"""
CIFAR-10 Custom CNN Training Script

Trains a CNN on CIFAR-10 and evaluates on a fixed stratified subset of 2,000
test images (200 per class) for fair comparison with GPT-4o Vision.

Features:
- Single training run with configurable epochs
- Epoch sweep mode: test multiple epoch values to find optimal
- Early stopping to prevent overfitting
- Comprehensive visualizations and metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os
import time
import argparse
from scipy.optimize import curve_fit
import copy
from datetime import datetime


# Manual implementations for demonstrating mathematical understanding

def manual_softmax(logits):
    """
    Manual implementation of softmax function.
    
    Mathematical Definition:
        σ(z)_i = exp(z_i) / Σ_j exp(z_j)
    
    Numerical Stability:
        We subtract max(z) from all elements to prevent exp() overflow.
        This doesn't change the result because:
        exp(z_i - c) / Σ exp(z_j - c) = exp(z_i)/exp(c) / (Σ exp(z_j)/exp(c))
                                      = exp(z_i) / Σ exp(z_j)
    
    Args:
        logits: Raw model outputs (batch_size, num_classes) as numpy array
    
    Returns:
        Probability distribution over classes
    """
    # Numerical stability: subtract max to prevent overflow
    z_stable = logits - np.max(logits, axis=-1, keepdims=True)
    
    # Compute exponentials
    exp_z = np.exp(z_stable)
    
    # Normalize to get probabilities
    probabilities = exp_z / np.sum(exp_z, axis=-1, keepdims=True)
    
    return probabilities


def manual_cross_entropy(probabilities, true_labels):
    """
    Manual implementation of cross-entropy loss.
    
    Mathematical Definition:
        L = -Σ_i y_i * log(p_i)
        
    For one-hot encoded labels with single correct class c:
        L = -log(p_c)
    
    Args:
        probabilities: Softmax outputs (batch_size, num_classes)
        true_labels: Integer class labels (batch_size,)
    
    Returns:
        Mean cross-entropy loss over the batch
    """
    batch_size = len(true_labels)
    
    # Small epsilon to prevent log(0)
    eps = 1e-15
    
    # Get probability of correct class for each sample
    correct_class_probs = probabilities[np.arange(batch_size), true_labels]
    
    # Compute negative log likelihood
    losses = -np.log(correct_class_probs + eps)
    
    # Return mean loss
    return np.mean(losses)


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """
    Manually compute confusion matrix.
    
    Mathematical Definition:
        CM[i,j] = count of samples with true label i and predicted label j
    
    The diagonal CM[i,i] represents correct predictions (True Positives for class i)
    Off-diagonal CM[i,j] (i≠j) represents misclassifications
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    
    return cm


def compute_metrics_from_confusion_matrix(cm, class_names=None):
    """
    Compute precision, recall, and F1-score from confusion matrix.
    
    Mathematical Definitions:
        For each class i:
        
        True Positives (TP_i)  = CM[i,i]
        False Positives (FP_i) = Σ_j CM[j,i] - CM[i,i] = column sum - diagonal
        False Negatives (FN_i) = Σ_j CM[i,j] - CM[i,i] = row sum - diagonal
        
        Precision_i = TP_i / (TP_i + FP_i)
            "Of all samples predicted as class i, what fraction are correct?"
            
        Recall_i = TP_i / (TP_i + FN_i)
            "Of all samples that ARE class i, what fraction did we find?"
            
        F1_i = 2 * (Precision_i * Recall_i) / (Precision_i + Recall_i)
            "Harmonic mean of precision and recall"
            
        Macro-Average = (1/K) * Σ_i metric_i
            "Simple average across classes (treats all classes equally)"
    
    Args:
        cm: Confusion matrix (num_classes, num_classes)
        class_names: Optional list of class names
    
    Returns:
        Dictionary containing per-class and aggregate metrics
    """
    num_classes = cm.shape[0]
    
    # Initialize arrays
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes)  # Number of samples per class
    
    for i in range(num_classes):
        # True Positives: diagonal element
        tp = cm[i, i]
        
        # False Positives: column sum minus diagonal (predicted i but wasn't i)
        fp = np.sum(cm[:, i]) - tp
        
        # False Negatives: row sum minus diagonal (was i but not predicted i)
        fn = np.sum(cm[i, :]) - tp
        
        # Support: total samples of class i
        support[i] = np.sum(cm[i, :])
        
        # Precision: TP / (TP + FP)
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1: Harmonic mean of precision and recall
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0.0
    
    # Compute aggregates
    metrics = {
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        },
        'macro_avg': {
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1': np.mean(f1)
        },
        'weighted_avg': {
            'precision': np.average(precision, weights=support),
            'recall': np.average(recall, weights=support),
            'f1': np.average(f1, weights=support)
        },
        'accuracy': np.trace(cm) / np.sum(cm)  # Sum of diagonal / total
    }
    
    return metrics


def print_classification_report(cm, class_names):
    """
    Print a formatted classification report similar to sklearn's.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    metrics = compute_metrics_from_confusion_matrix(cm, class_names)
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT (Computed from Confusion Matrix)")
    print("=" * 70)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)
    
    for i, name in enumerate(class_names):
        print(f"{name:<12} {metrics['per_class']['precision'][i]:>10.4f} "
              f"{metrics['per_class']['recall'][i]:>10.4f} "
              f"{metrics['per_class']['f1'][i]:>10.4f} "
              f"{int(metrics['per_class']['support'][i]):>10}")
    
    print("-" * 70)
    total_support = int(np.sum(metrics['per_class']['support']))
    
    print(f"{'Macro Avg':<12} {metrics['macro_avg']['precision']:>10.4f} "
          f"{metrics['macro_avg']['recall']:>10.4f} "
          f"{metrics['macro_avg']['f1']:>10.4f} "
          f"{total_support:>10}")
    
    print(f"{'Weighted Avg':<12} {metrics['weighted_avg']['precision']:>10.4f} "
          f"{metrics['weighted_avg']['recall']:>10.4f} "
          f"{metrics['weighted_avg']['f1']:>10.4f} "
          f"{total_support:>10}")
    
    print("-" * 70)
    print(f"{'Accuracy':<12} {'':<10} {'':<10} {metrics['accuracy']:>10.4f} "
          f"{total_support:>10}")
    print("=" * 70)
    
    return metrics

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(0.2)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class CustomCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=224):
        super(CustomCNN, self).__init__()
        self.input_size = input_size
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256)
        )
        
        self.fc1 = nn.Linear(256 * 14 * 14, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

RESULTS_DIR = None


def create_stratified_test_subset(test_dataset, num_samples=2000, seed=42):
    """
    Load stratified subset of test data from stratified_subset_2000.json
    
    This ensures fair comparison between CNN and GPT-4o by using
    the exact same test images for both models.
    
    Args:
        test_dataset: CIFAR-10 test dataset
        num_samples: Total number of samples (must be divisible by 10)
        seed: Random seed for reproducibility
    
    Returns:
        List of indices for the stratified subset
    """
    # Use the shared stratified_subset_2000.json from root directory
    indices_file = 'stratified_subset_2000.json'
    
    if os.path.exists(indices_file):
        print(f"Loading stratified test indices from {indices_file}")
        with open(indices_file, 'r') as f:
            data = json.load(f)
            # Handle both formats: plain list or dict with "indices" key
            if isinstance(data, dict) and 'indices' in data:
                return data['indices']
            return data
    
    np.random.seed(seed)
    samples_per_class = num_samples // 10
    
    # Group indices by class
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        class_indices[label].append(idx)
    
    # Sample from each class
    stratified_indices = []
    for class_idx in range(10):
        samples = np.random.choice(class_indices[class_idx], samples_per_class, replace=False)
        stratified_indices.extend(samples.tolist())
    
    np.random.shuffle(stratified_indices)
    stratified_indices = [int(i) for i in stratified_indices]
    
    print(f"Created stratified subset: {num_samples} images ({samples_per_class} per class)")
    
    return stratified_indices


def _upscale_image(img, size):
    from PIL import Image
    return img.resize((size, size), Image.BILINEAR)

def _upscale_to_size(size):
    from PIL import Image
    def upscale_fn(img):
        return img.resize((size, size), Image.BILINEAR)
    return upscale_fn

def get_data_loaders(batch_size=128, input_size=224):
    upscale_transform = transforms.Lambda(_upscale_to_size(input_size))
    
    train_transform = transforms.Compose([
        upscale_transform,
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(input_size, padding=input_size//8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.1)
    ])
    
    test_transform = transforms.Compose([
        upscale_transform,
        transforms.ToTensor()
    ])
    
    print(f"Loading CIFAR-10 (upscaled to {input_size}×{input_size} using bilinear interpolation, same as GPT-4o)...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"  Input size: {input_size}×{input_size} (upscaled from 32×32 using Image.BILINEAR)")
    return train_loader, test_dataset


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / total, 100. * correct / total


def evaluate(model, test_dataset, indices, criterion, device):
    """Evaluate on stratified test subset"""
    model.eval()
    running_loss = 0.0
    correct = 0
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Evaluating', leave=False):
            img, label = test_dataset[idx]
            # Image is already upscaled in test_dataset transform (if using 224 version)
            img_normalized = normalize(img).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label]).to(device)
            
            outputs = model(img_normalized)
            running_loss += criterion(outputs, label_tensor).item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(label_tensor).sum().item()
    
    return running_loss / len(indices), 100. * correct / len(indices)


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def compute_metrics(cm):
    """Compute precision, recall, F1 from confusion matrix"""
    num_classes = cm.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.sum(cm, axis=1)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0.0
    
    return {
        'per_class': {'precision': precision, 'recall': recall, 'f1': f1, 'support': support},
        'macro_avg': {'precision': np.mean(precision), 'recall': np.mean(recall), 'f1': np.mean(f1)},
        'weighted_avg': {
            'precision': np.average(precision, weights=support),
            'recall': np.average(recall, weights=support),
            'f1': np.average(f1, weights=support)
        },
        'accuracy': np.trace(cm) / np.sum(cm)
    }


def print_classification_report(cm, class_names):
    """Print formatted classification report"""
    metrics = compute_metrics(cm)
    
    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    
    for i, name in enumerate(class_names):
        print(f"{name:<12} {metrics['per_class']['precision'][i]:>10.4f} "
              f"{metrics['per_class']['recall'][i]:>10.4f} "
              f"{metrics['per_class']['f1'][i]:>10.4f} "
              f"{int(metrics['per_class']['support'][i]):>10}")
    
    print("-" * 55)
    total = int(np.sum(metrics['per_class']['support']))
    print(f"{'Macro Avg':<12} {metrics['macro_avg']['precision']:>10.4f} "
          f"{metrics['macro_avg']['recall']:>10.4f} {metrics['macro_avg']['f1']:>10.4f} {total:>10}")
    print(f"{'Accuracy':<12} {'':<10} {'':<10} {metrics['accuracy']:>10.4f} {total:>10}")
    
    return metrics


def final_evaluation(model, test_dataset, indices, device):
    """Final evaluation with confusion matrix and metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    print(f"\nEvaluating on {len(indices)} test images...")
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Final evaluation'):
            img, label = test_dataset[idx]
            img_normalized = normalize(img).unsqueeze(0).to(device)
            
            outputs = model(img_normalized)
            _, predicted = outputs.max(1)
            
            all_predictions.append(predicted.item())
            all_labels.append(label)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    cm = compute_confusion_matrix(all_labels, all_predictions)
    metrics = print_classification_report(cm, CLASSES)
    
    # Most confused pair
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    most_confused = np.unravel_index(np.argmax(cm_no_diag), cm.shape)
    print(f"\nMost confused: {CLASSES[most_confused[0]]} -> {CLASSES[most_confused[1]]} "
          f"({cm_no_diag[most_confused]} errors)")
    
    plot_confusion_matrix(cm, CLASSES)
    
    return {'confusion_matrix': cm, 'metrics': metrics, 'predictions': all_predictions, 'labels': all_labels}


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix with default path in results directory"""
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    """
    Visualize the confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True', xlabel='Predicted', title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    thresh = cm_norm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                   color='white' if cm_norm[i, j] > thresh else 'black', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close()


def train_model(model, train_loader, test_dataset, test_indices, epochs=100, lr=0.001, 
                early_stopping=False, patience=5, flatline_patience=20, verbose=True):
    """
    Train with evaluation on stratified test subset
    
    Args:
        model: The CNN model to train
        train_loader: DataLoader for training data
        test_dataset: Test dataset
        test_indices: Indices for stratified test subset
        epochs: Maximum number of epochs to train (increased default to detect overfitting)
        lr: Learning rate
        early_stopping: Whether to enable early stopping
        patience: Number of epochs without improvement before stopping
        flatline_patience: Number of epochs with no accuracy change before stopping (default: 20)
        verbose: Whether to print progress
    
    Returns:
        history: Dictionary with training metrics
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'epoch_times': []}
    best_acc = 0.0
    best_model_state = None
    epochs_without_improvement = 0
    epochs_without_change = 0
    last_test_acc = None
    actual_epochs = 0
    stopped_reason = None
    
    if verbose:
        print(f"\nTraining for up to {epochs} epochs...")
        if early_stopping:
            print(f"Early stopping enabled (patience={patience})")
        print(f"Flatline detection enabled (stops if accuracy unchanged for {flatline_patience} epochs)")
    
    training_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        if verbose:
            print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_dataset, test_indices, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start_time
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['epoch_times'].append(epoch_time)
        actual_epochs = epoch + 1
        
        if verbose:
            print(f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        # Check for flatline (no change in accuracy)
        if last_test_acc is not None:
            # Consider unchanged if difference is less than 0.01% (essentially flat)
            if abs(test_acc - last_test_acc) < 0.01:
                epochs_without_change += 1
            else:
                epochs_without_change = 0
        last_test_acc = test_acc
        
        # Check for improvement
        if test_acc > best_acc:
            best_acc = test_acc
            epochs_without_improvement = 0
            best_model_state = copy.deepcopy(model.state_dict())
            model_path = os.path.join(RESULTS_DIR, 'best_cnn_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_subset_size': len(test_indices),
            }, model_path)
            if verbose:
                print(f"  Best model saved! (Test Acc: {test_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
        
        # Check early stopping conditions
        if early_stopping and epochs_without_improvement >= patience:
            stopped_reason = f"No improvement for {patience} epochs"
            if verbose:
                print(f"\nEarly stopping triggered. {stopped_reason}.")
                print(f"  Stopped at epoch {epoch+1}, best accuracy was {best_acc:.2f}%")
            break
        
        # Check flatline condition (always enabled, not just when early_stopping is True)
        if epochs_without_change >= flatline_patience:
            stopped_reason = f"Accuracy flatlined for {flatline_patience} epochs"
            if verbose:
                print(f"\nFlatline detection triggered. {stopped_reason}.")
                print(f"  Stopped at epoch {epoch+1}, accuracy: {test_acc:.2f}%")
            break
    
    total_training_time = time.time() - training_start_time
    history['total_training_time'] = total_training_time
    history['actual_epochs'] = actual_epochs
    history['best_acc'] = best_acc
    history['early_stopped'] = stopped_reason is not None
    history['stopped_reason'] = stopped_reason
    
    # Restore best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if verbose:
        print(f"\nBest accuracy: {best_acc:.2f}%")
        print(f"Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
        if history['early_stopped']:
            print(f"Training stopped early at epoch {actual_epochs}: {stopped_reason}")
    
    return history


def run_epoch_sweep(train_loader, test_dataset, test_indices, epoch_list, lr=0.001, 
                   early_stopping=False, patience=5, flatline_patience=20, input_size=224):
    """
    Run training with different epoch values to find optimal training length.
    
    This is useful for:
    1. Finding the point of diminishing returns
    2. Detecting when overfitting begins
    3. Optimizing training time vs accuracy trade-off
    
    Args:
        train_loader: DataLoader for training data
        test_dataset: Test dataset
        test_indices: Indices for stratified test subset
        epoch_list: List of epoch values to test (e.g., [5, 10, 20, 30, 50])
        lr: Learning rate
        early_stopping: Whether to enable early stopping
        patience: Early stopping patience
        flatline_patience: Number of epochs with no accuracy change before stopping
    
    Returns:
        sweep_results: Dictionary with results for each epoch value
    """
    print("\n" + "=" * 70)
    print("EPOCH SWEEP EXPERIMENT")
    print("=" * 70)
    print(f"Testing epoch values: {epoch_list}")
    print(f"Training {len(epoch_list)} separate models to compare accuracy")
    print("=" * 70)
    
    sweep_results = {
        'epoch_values': epoch_list,
        'accuracies': [],
        'training_times': [],
        'best_epochs': [],
        'histories': [],
    }
    
    for i, epochs in enumerate(epoch_list):
        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(epoch_list)}] Training with {epochs} epochs...")
        print('─' * 70)
        
        # Create fresh model for each experiment
        model = CustomCNN(input_size=input_size).to(device)
        
        history = train_model(
            model, train_loader, test_dataset, test_indices,
            epochs=epochs, lr=lr, early_stopping=early_stopping, 
            patience=patience, flatline_patience=flatline_patience, verbose=True
        )
        
        sweep_results['accuracies'].append(history['best_acc'])
        sweep_results['training_times'].append(history['total_training_time'])
        sweep_results['best_epochs'].append(history['test_acc'].index(max(history['test_acc'])) + 1)
        sweep_results['histories'].append({
            'train_acc': history['train_acc'],
            'test_acc': history['test_acc'],
            'train_loss': history['train_loss'],
            'test_loss': history['test_loss'],
        })
    
    # Find optimal epoch count
    best_idx = np.argmax(sweep_results['accuracies'])
    sweep_results['optimal_epochs'] = epoch_list[best_idx]
    sweep_results['optimal_accuracy'] = sweep_results['accuracies'][best_idx]
    
    # Calculate efficiency (accuracy per training minute)
    sweep_results['efficiency'] = [
        acc / (time_s / 60) if time_s > 0 else 0 
        for acc, time_s in zip(sweep_results['accuracies'], sweep_results['training_times'])
    ]
    
    print("\n" + "=" * 70)
    print("EPOCH SWEEP RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Epochs':<10} {'Accuracy':>12} {'Time (min)':>12} {'Eff (%/min)':>12}")
    print("-" * 50)
    for i, epochs in enumerate(epoch_list):
        marker = " *" if i == best_idx else ""
        print(f"{epochs:<10} {sweep_results['accuracies'][i]:>11.2f}% "
              f"{sweep_results['training_times'][i]/60:>11.1f} "
              f"{sweep_results['efficiency'][i]:>11.2f}{marker}")
    print("-" * 50)
    print(f"\nOPTIMAL: {sweep_results['optimal_epochs']} epochs -> {sweep_results['optimal_accuracy']:.2f}% accuracy")
    
    # Plot epoch sweep comparison
    plot_epoch_sweep_comparison(sweep_results)
    
    # Save sweep results
    sweep_results_path = os.path.join(RESULTS_DIR, 'epoch_sweep_results.json')
    # Convert numpy types for JSON serialization
    serializable_results = {
        'epoch_values': [int(e) for e in sweep_results['epoch_values']],
        'accuracies': [float(a) for a in sweep_results['accuracies']],
        'training_times': [float(t) for t in sweep_results['training_times']],
        'best_epochs': [int(e) for e in sweep_results['best_epochs']],
        'optimal_epochs': int(sweep_results['optimal_epochs']),
        'optimal_accuracy': float(sweep_results['optimal_accuracy']),
        'efficiency': [float(e) for e in sweep_results['efficiency']],
    }
    with open(sweep_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nSweep results saved to {sweep_results_path}")
    
    return sweep_results


def plot_epoch_sweep_comparison(sweep_results):
    """
    Create visualization comparing different epoch training runs.
    
    This helps identify:
    1. Optimal epoch count for best accuracy
    2. Diminishing returns point
    3. Time vs accuracy trade-off
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    epochs = sweep_results['epoch_values']
    accuracies = sweep_results['accuracies']
    times = [t/60 for t in sweep_results['training_times']]  # Convert to minutes
    best_idx = np.argmax(accuracies)
    
    # Color scheme
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(epochs)))
    
    # Plot 1: Accuracy vs Epochs
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(len(epochs)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
    bars[best_idx].set_color('#FFD700')  # Gold for best
    bars[best_idx].set_edgecolor('#B8860B')
    bars[best_idx].set_linewidth(3)
    
    ax1.set_xticks(range(len(epochs)))
    ax1.set_xticklabels([str(e) for e in epochs])
    ax1.set_xlabel('Number of Epochs', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Training Epochs', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        label = f'{acc:.1f}%'
        if i == best_idx:
            label += '\n(best)'
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylim([min(accuracies) - 5, max(accuracies) + 8])
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Training Time vs Epochs
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(len(epochs)), times, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(epochs)))
    ax2.set_xticklabels([str(e) for e in epochs])
    ax2.set_xlabel('Number of Epochs', fontsize=12)
    ax2.set_ylabel('Training Time (minutes)', fontsize=12)
    ax2.set_title('Training Time vs Epochs', fontsize=14, fontweight='bold')
    
    for i, (t, e) in enumerate(zip(times, epochs)):
        ax2.text(i, t + 0.2, f'{t:.1f}m', ha='center', va='bottom', fontsize=10)
    
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Accuracy progression line plot
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(epochs, accuracies, 'o-', color='#2E86AB', linewidth=2.5, markersize=10)
    ax3.scatter([epochs[best_idx]], [accuracies[best_idx]], color='#FFD700', s=200, 
                zorder=5, edgecolor='black', linewidth=2, label=f'Best: {epochs[best_idx]} epochs')
    
    # Add trend line
    if len(epochs) >= 3:
        z = np.polyfit(epochs, accuracies, 2)
        p = np.poly1d(z)
        x_smooth = np.linspace(min(epochs), max(epochs), 100)
        ax3.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.5, label='Trend (quadratic fit)')
    
    ax3.set_xlabel('Number of Epochs', fontsize=12)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax3.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
    ax3.legend(loc='lower right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training efficiency (accuracy per minute)
    ax4 = fig.add_subplot(gs[1, 1])
    efficiency = sweep_results['efficiency']
    bars = ax4.bar(range(len(epochs)), efficiency, color=colors, edgecolor='black', linewidth=1.5)
    
    best_eff_idx = np.argmax(efficiency)
    bars[best_eff_idx].set_color('#90EE90')  # Light green for most efficient
    bars[best_eff_idx].set_edgecolor('#228B22')
    bars[best_eff_idx].set_linewidth(3)
    
    ax4.set_xticks(range(len(epochs)))
    ax4.set_xticklabels([str(e) for e in epochs])
    ax4.set_xlabel('Number of Epochs', fontsize=12)
    ax4.set_ylabel('Efficiency (% accuracy / minute)', fontsize=12)
    ax4.set_title('Training Efficiency', fontsize=14, fontweight='bold')
    
    for i, (bar, eff) in enumerate(zip(bars, efficiency)):
        label = f'{eff:.1f}'
        if i == best_eff_idx:
            label += '\n(best)'
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                label, ha='center', va='bottom', fontsize=10)
    
    ax4.grid(axis='y', alpha=0.3)
    
    # Add summary text box
    summary = (f"Best Accuracy: {max(accuracies):.2f}% @ {epochs[best_idx]} epochs\n"
               f"Best Efficiency: {max(efficiency):.2f} %/min @ {epochs[best_eff_idx]} epochs\n"
               f"Recommendation: Use {epochs[best_idx]} epochs for best results")
    
    fig.text(0.5, 0.02, summary, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Epoch Sweep Experiment Results', fontsize=16, fontweight='bold', y=0.98)
    
    sweep_plot_path = os.path.join(RESULTS_DIR, 'epoch_sweep_comparison.png')
    plt.savefig(sweep_plot_path, dpi=150, bbox_inches='tight')
    print(f"\nEpoch sweep comparison saved to {sweep_plot_path}")
    plt.close()


def plot_training_history(history, epochs=20):
    """
    Plot training history with mathematical convergence analysis.
    
    Mathematical Indicators:
    1. Loss Decay Rate: L(t) ≈ L₀·e^(-αt) - exponential convergence
    2. Convergence Rate: |acc(t) - acc(t-1)| - rate of improvement
    3. Overfitting Gap: train_acc - test_acc - generalization gap
    4. Learning Rate Schedule: step decay visualization
    """
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    epochs_list = list(range(1, len(history['train_loss']) + 1))
    
    # Plot 1: Loss curves with exponential fit
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs_list, history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax1.plot(epochs_list, history['test_loss'], label='Test Loss', marker='s', linewidth=2)
    
    # Fit exponential decay: L(t) = a·e^(-bt) + c
    try:
        def exp_decay(t, a, b, c):
            return a * np.exp(-b * np.array(t)) + c
        
        # Fit to test loss (more stable)
        if len(history['test_loss']) >= 5:
            popt, _ = curve_fit(exp_decay, epochs_list, history['test_loss'], 
                              p0=[history['test_loss'][0], 0.1, history['test_loss'][-1]])
            fit_epochs = np.linspace(1, len(history['test_loss']), 100)
            fit_loss = exp_decay(fit_epochs, *popt)
            ax1.plot(fit_epochs, fit_loss, '--', color='red', alpha=0.5, 
                    label=f'Exp Fit: L(t)={popt[0]:.3f}·e^(-{popt[1]:.3f}t)+{popt[2]:.3f}')
            ax1.text(0.05, 0.95, f'Decay Rate α = {popt[1]:.4f}', 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except:
        pass
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Convergence: L(t) ≈ L₀·e^(-αt)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_list, history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    ax2.plot(epochs_list, history['test_acc'], label='Test Acc', marker='s', linewidth=2)
    
    # Add GPT-4o result line (96.8%)
    GPT4O_ACCURACY = 96.8
    ax2.axhline(GPT4O_ACCURACY, color='red', linestyle='-', linewidth=2.5, 
                label=f'GPT-4o Vision ({GPT4O_ACCURACY}%)', alpha=0.8)
    
    # Add GPT-4o target range (85-95%) as background
    ax2.axhspan(85, 95, alpha=0.15, color='green', label='GPT-4o Range (85-95%)')
    ax2.axhline(85, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax2.axhline(95, color='green', linestyle='--', alpha=0.3, linewidth=1)
    
    # Final accuracy annotation
    final_acc = history['test_acc'][-1]
    ax2.annotate(f'Final: {final_acc:.2f}%', 
                xy=(len(history['test_acc']), final_acc),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add gap to GPT-4o annotation
    gap_to_gpt4o = GPT4O_ACCURACY - final_acc
    if gap_to_gpt4o > 0:
        ax2.annotate(f'Gap to GPT-4o: {gap_to_gpt4o:.2f}%', 
                    xy=(len(history['test_acc']) * 0.7, GPT4O_ACCURACY - 2),
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
                    fontsize=9)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy Convergence (vs GPT-4o)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Convergence rate
    ax3 = fig.add_subplot(gs[1, 0])
    if len(history['test_acc']) > 1:
        convergence_rate = np.abs(np.diff(history['test_acc']))
        ax3.plot(epochs_list[1:], convergence_rate, marker='o', color='purple', linewidth=2)
        ax3.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Convergence threshold (0.1%)')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('|Δ Accuracy| (%)', fontsize=11)
        ax3.set_title('Convergence Rate: |acc(t) - acc(t-1)|', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # Plot 4: Overfitting gap (train_acc - test_acc)
    ax4 = fig.add_subplot(gs[1, 1])
    overfitting_gap = np.array(history['train_acc']) - np.array(history['test_acc'])
    ax4.plot(epochs_list, overfitting_gap, marker='o', color='orange', linewidth=2)
    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.fill_between(epochs_list, 0, overfitting_gap, alpha=0.3, color='orange')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Train Acc - Test Acc (%)', fontsize=11)
    ax4.set_title('Generalization Gap (Overfitting Indicator)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Learning rate schedule
    ax5 = fig.add_subplot(gs[2, 0])
    # Calculate LR schedule: lr * (gamma ^ floor(epoch / step_size))
    initial_lr = 0.001
    step_size = 10
    gamma = 0.1
    lr_schedule = [initial_lr * (gamma ** (epoch // step_size)) for epoch in epochs_list]
    ax5.plot(epochs_list, lr_schedule, marker='s', color='blue', linewidth=2)
    ax5.set_xlabel('Epoch', fontsize=11)
    ax5.set_ylabel('Learning Rate', fontsize=11)
    ax5.set_title(f'LR Schedule: lr(t) = lr₀ · γ^⌊t/{step_size}⌋, γ={gamma}', 
                  fontsize=12, fontweight='bold')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
    # GPT-4o accuracy for comparison
    GPT4O_ACCURACY = 96.8
    
    # Calculate statistics
    final_train_loss = history['train_loss'][-1]
    final_test_loss = history['test_loss'][-1]
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    best_test_acc = max(history['test_acc'])
    best_epoch = history['test_acc'].index(best_test_acc) + 1
    
    # Loss improvement
    loss_improvement = (history['test_loss'][0] - final_test_loss) / history['test_loss'][0] * 100
    
    # Accuracy improvement rate (last 5 epochs)
    if len(history['test_acc']) >= 5:
        recent_improvement = history['test_acc'][-1] - history['test_acc'][-5]
    else:
        recent_improvement = 0
    
    summary_text = f"""
MATHEMATICAL CONVERGENCE ANALYSIS
{'='*50}

Loss Function:
  Initial Loss: {history['test_loss'][0]:.4f}
  Final Loss:   {final_test_loss:.4f}
  Improvement:  {loss_improvement:.2f}%
  
Accuracy Function:
  Final Accuracy: {final_test_acc:.2f}%
  Best Accuracy:  {best_test_acc:.2f}% (Epoch {best_epoch})
  Recent Δ (last 5): {recent_improvement:+.2f}%
  
Generalization:
  Train Acc: {final_train_acc:.2f}%
  Test Acc:  {final_test_acc:.2f}%
  Gap:       {overfitting_gap[-1]:.2f}%
  
Convergence Status:
  {'Converged' if len(history['test_acc']) > 5 and recent_improvement < 0.5 else 'Still improving'}
  
Target Comparison:
  Current: {final_test_acc:.2f}%
  GPT-4o:  {GPT4O_ACCURACY}%
  Gap:     {GPT4O_ACCURACY - final_test_acc:.2f}% to reach GPT-4o
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Mathematical Convergence Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    history_path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    print(f"Training history with mathematical analysis saved to {history_path}")
    plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CNN on CIFAR-10')
    parser.add_argument('--epochs', type=str, default='100',
                        help='Number of epochs (single value or comma-separated for sweep, default: 100)')
    parser.add_argument('--sweep', action='store_true',
                        help='Enable epoch sweep mode')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    parser.add_argument('--flatline-patience', type=int, default=20,
                        help='Flatline detection patience - stops if accuracy unchanged (default: 20)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001 for 224×224)')
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start_time = time.time()
    
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EARLY_STOPPING = args.early_stopping
    PATIENCE = args.patience
    NUM_TEST_SAMPLES = 2000  # Stratified subset for fair comparison with GPT-4o
    
    # Parse epochs
    if args.sweep:
        EPOCH_LIST = [int(e.strip()) for e in args.epochs.split(',')]
        print(f"\nEPOCH SWEEP MODE")
        print(f"   Testing epochs: {EPOCH_LIST}")
    else:
        EPOCHS = int(args.epochs.split(',')[0])
    
    print("\nHYPERPARAMETERS (with mathematical justification):")
    print(f"   Batch Size: {BATCH_SIZE}")
    print("      - Larger batches -> more stable gradient estimates")
    print("      - Trade-off between memory usage and gradient variance")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print("      - Controls step size: θ = θ - lr * ∇L")
    print("      - Too high -> divergence, too low -> slow convergence")
    if args.sweep:
        print(f"   Epoch Values: {EPOCH_LIST}")
    else:
        print(f"   Epochs: {EPOCHS}")
    print("      - One epoch = one pass through all training data")
    if EARLY_STOPPING:
        print(f"   Early Stopping: ENABLED (patience={PATIENCE})")
    
    global RESULTS_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = f'results_cnn_224_{timestamp}'
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\nResults will be saved to: {RESULTS_DIR}/")
    
    # Load data (upscaled to 224×224 to match GPT-4o)
    INPUT_SIZE = 224  # Match GPT-4o Vision input size
    train_loader, test_dataset = get_data_loaders(BATCH_SIZE, input_size=INPUT_SIZE)
    test_indices = create_stratified_test_subset(test_dataset)
    
    # Epoch sweep mode: train multiple models with different epoch counts
    if args.sweep:
        sweep_results = run_epoch_sweep(
            train_loader, test_dataset, test_indices,
            epoch_list=EPOCH_LIST, lr=LEARNING_RATE,
            early_stopping=EARLY_STOPPING, patience=PATIENCE,
            flatline_patience=args.flatline_patience, input_size=INPUT_SIZE
        )
        
        total_pipeline_time = time.time() - pipeline_start_time
        print("\n" + "=" * 70)
        print("EPOCH SWEEP COMPLETE!")
        print("=" * 70)
        print(f"\nOPTIMAL CONFIGURATION:")
        print(f"   Epochs: {sweep_results['optimal_epochs']}")
        print(f"   Accuracy: {sweep_results['optimal_accuracy']:.2f}%")
        print(f"\nGenerated files (saved to {RESULTS_DIR}/):")
        print(f"   • epoch_sweep_comparison.png  - Visual comparison of all runs")
        print(f"   • epoch_sweep_results.json    - Detailed sweep results")
        print(f"   • best_cnn_model.pth          - Best model weights")
        print(f"\nTotal time: {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f} min)")
        return
    
    # Single training mode
    INPUT_SIZE = 224  # Match GPT-4o Vision input size
    model = CustomCNN(input_size=INPUT_SIZE).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {params:,}")
    print(f"Input size: {INPUT_SIZE}×{INPUT_SIZE} (upscaled from 32×32 using bilinear interpolation, same as GPT-4o)")
    
    history = train_model(
        model, train_loader, test_dataset, test_indices, 
        epochs=EPOCHS, lr=LEARNING_RATE,
        early_stopping=EARLY_STOPPING, patience=PATIENCE,
        flatline_patience=args.flatline_patience
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, epochs=EPOCHS)
    
    # Save training history
    history_json_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_json_path}")
    
    # Save timing data to separate CSV file (easy to analyze)
    timing_csv_path = os.path.join(RESULTS_DIR, 'epoch_timing.csv')
    with open(timing_csv_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc,epoch_time_sec,cumulative_time_sec\n")
        cumulative = 0
        for i in range(len(history['epoch_times'])):
            cumulative += history['epoch_times'][i]
            f.write(f"{i+1},{history['train_loss'][i]:.6f},{history['train_acc'][i]:.2f},"
                    f"{history['test_loss'][i]:.6f},{history['test_acc'][i]:.2f},"
                    f"{history['epoch_times'][i]:.2f},{cumulative:.2f}\n")
    print(f"Epoch timing saved to {timing_csv_path}")
    
    # Final evaluation with best model
    print("\nLoading best model for final evaluation...")
    model_path = os.path.join(RESULTS_DIR, 'best_cnn_model.pth')
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model (epoch {checkpoint['epoch'] + 1}, {checkpoint['test_acc']:.2f}%)")
    
    eval_start_time = time.time()
    results = final_evaluation(model, test_dataset, test_indices, device)
    eval_time = time.time() - eval_start_time
    
    # Calculate total pipeline time
    total_pipeline_time = time.time() - pipeline_start_time
    
    # Create evaluation summary with timing info
    eval_summary = {
        'accuracy': results['metrics']['accuracy'],
        'macro_f1': results['metrics']['macro_avg']['f1'],
        'epochs': history.get('actual_epochs', EPOCHS),
        'early_stopped': history.get('early_stopped', False),
        'timing': {
            'total_training_time_seconds': history.get('total_training_time', 0),
            'total_training_time_minutes': history.get('total_training_time', 0) / 60,
            'avg_epoch_time_seconds': np.mean(history.get('epoch_times', [0])),
            'final_evaluation_time_seconds': eval_time,
            'total_pipeline_time_seconds': total_pipeline_time,
            'total_pipeline_time_minutes': total_pipeline_time / 60,
        }
    }
    
    eval_results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    print(f"\nEvaluation results saved to {eval_results_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated files (saved to {RESULTS_DIR}/):")
    print(f"   • best_cnn_model.pth           - Trained model weights")
    print(f"   • training_history.png         - Loss/accuracy plots")
    print(f"   • training_history.json        - Training metrics data")
    print(f"   • epoch_timing.csv             - Per-epoch timing data")
    print(f"   • confusion_matrix.png         - Confusion matrix visualization")
    print(f"   • evaluation_results.json      - Final evaluation metrics")
    print(f"\nEvaluation used {NUM_TEST_SAMPLES} stratified test images (same as GPT-4o)")
    
    print(f"\n" + "-" * 70)
    print("TIMING SUMMARY:")
    print(f"   • Training time:      {history.get('total_training_time', 0):.1f}s ({history.get('total_training_time', 0)/60:.1f} min)")
    print(f"   • Avg epoch time:     {np.mean(history.get('epoch_times', [0])):.1f}s")
    print(f"   • Final evaluation:   {eval_time:.1f}s")
    print(f"   • Total pipeline:     {total_pipeline_time:.1f}s ({total_pipeline_time/60:.1f} min)")
    if history.get('early_stopped'):
        print(f"   • Early stopping:     Triggered at epoch {history.get('actual_epochs')}")
    print("-" * 70)


if __name__ == "__main__":
    main()
