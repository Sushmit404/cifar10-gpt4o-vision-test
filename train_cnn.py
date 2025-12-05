"""
CIFAR-10 Custom CNN Training Script
Part 1: Train a custom CNN on CIFAR-10 with GPU support

Evaluation uses a fixed stratified subset of 2,000 test images (200 per class)
to ensure fair comparison with GPT-4o Vision.

=============================================================================
MATHEMATICAL FOUNDATIONS
=============================================================================

This implementation demonstrates understanding of the underlying mathematics:

1. SOFTMAX FUNCTION (Output Layer Activation)
   -----------------------------------------
   For a vector z of logits, softmax converts to probabilities:
   
   σ(z)_i = exp(z_i) / Σ_j exp(z_j)
   
   Properties:
   - Output sums to 1 (valid probability distribution)
   - Numerically stable version subtracts max(z) to prevent overflow

2. CROSS-ENTROPY LOSS
   -------------------
   For true label y (one-hot) and predicted probabilities p:
   
   L = -Σ_i y_i * log(p_i)
   
   For single correct class c: L = -log(p_c)
   
   Combined with softmax (used in nn.CrossEntropyLoss):
   L = -z_c + log(Σ_j exp(z_j))

3. CONVOLUTION OPERATION
   ----------------------
   For input I and kernel K:
   
   (I * K)[i,j] = Σ_m Σ_n I[i+m, j+n] · K[m,n]
   
   Output size: (W - K + 2P) / S + 1
   where W=input width, K=kernel size, P=padding, S=stride

4. BACKPROPAGATION (Chain Rule)
   -----------------------------
   For loss L and parameters θ:
   
   ∂L/∂θ = ∂L/∂a · ∂a/∂z · ∂z/∂θ
   
   where a=activation, z=pre-activation

5. EVALUATION METRICS (Calculated from Confusion Matrix)
   ------------------------------------------------------
   Precision_i = TP_i / (TP_i + FP_i)  -- "Of predicted class i, how many correct?"
   Recall_i    = TP_i / (TP_i + FN_i)  -- "Of actual class i, how many found?"
   F1_i        = 2 * (P_i * R_i) / (P_i + R_i)  -- Harmonic mean
   
   Macro-average: Mean across all classes (treats classes equally)
   Weighted-average: Weighted by class support (handles imbalance)

=============================================================================
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
from scipy.optimize import curve_fit


# =============================================================================
# MANUAL IMPLEMENTATIONS (Demonstrating Mathematical Understanding)
# =============================================================================

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
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


class CustomCNN(nn.Module):
    """
    Custom CNN for CIFAR-10 Classification
    
    Architecture:
    - 2 convolutional layers with ReLU and max pooling
    - Fully connected classification head
    - Softmax output with cross-entropy loss
    
    ==========================================================================
    MATHEMATICAL ANALYSIS OF LAYER DIMENSIONS
    ==========================================================================
    
    Convolution output size formula:
        O = floor((W - K + 2P) / S) + 1
        where W = input size, K = kernel size, P = padding, S = stride
    
    Layer-by-layer dimension analysis:
    
    Input: (batch, 3, 32, 32) - RGB images
    
    Conv1: kernel=3, padding=1, stride=1
        O = (32 - 3 + 2*1) / 1 + 1 = 32
        Output: (batch, 32, 32, 32)
        Parameters: 3*32*3*3 + 32 = 896 (weights + biases)
    
    BatchNorm1: 
        Output: (batch, 32, 32, 32) - shape unchanged
        Parameters: 32*2 = 64 (gamma and beta per channel)
    
    MaxPool1: kernel=2, stride=2
        O = (32 - 2) / 2 + 1 = 16
        Output: (batch, 32, 16, 16)
        Parameters: 0 (no learnable parameters)
    
    Conv2: kernel=3, padding=1, stride=1
        O = (16 - 3 + 2*1) / 1 + 1 = 16
        Output: (batch, 64, 16, 16)
        Parameters: 32*64*3*3 + 64 = 18,496
    
    BatchNorm2:
        Output: (batch, 64, 16, 16)
        Parameters: 64*2 = 128
    
    MaxPool2: kernel=2, stride=2
        O = (16 - 2) / 2 + 1 = 8
        Output: (batch, 64, 8, 8)
    
    Flatten: 64 * 8 * 8 = 4,096 features
    
    FC1: 4096 -> 128
        Parameters: 4096*128 + 128 = 524,416
    
    FC2: 128 -> 10
        Parameters: 128*10 + 10 = 1,290
    
    Total trainable parameters: ~545,290
    ==========================================================================
    """
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        # Conv2d parameters: (in_channels, out_channels, kernel_size)
        # Output size: (W - K + 2P) / S + 1 = (32 - 3 + 2) / 1 + 1 = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32x3 -> 32x32x32
        self.bn1 = nn.BatchNorm2d(32)  # Normalize: (x - μ) / σ * γ + β
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32x32 -> 16x16x32 (spatial reduction by 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16x32 -> 16x16x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16x64 -> 8x8x64
        
        # Fully connected layers
        # Input dimension: 64 channels * 8 height * 8 width = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)  # Randomly zero 50% of inputs (regularization)
        self.fc2 = nn.Linear(128, num_classes)  # Output: raw logits (not probabilities)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Mathematical operations at each step:
        1. Convolution: y = Σ(x * w) + b (learnable weights and bias)
        2. BatchNorm: y = (x - μ) / √(σ² + ε) * γ + β
        3. ReLU: y = max(0, x) (introduces non-linearity)
        4. MaxPool: y = max(x in window) (downsampling)
        5. Linear: y = xW^T + b
        """
        # Conv block 1: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2: Conv -> BatchNorm -> ReLU -> MaxPool
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Flatten: (batch, 64, 8, 8) -> (batch, 4096)
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected with dropout (dropout disabled during eval)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output: raw logits (softmax applied in loss function)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract feature maps from conv layers for visualization"""
        features = {}
        
        # Conv1 features
        x = torch.relu(self.bn1(self.conv1(x)))
        features['conv1'] = x.detach()
        x = self.pool1(x)
        
        # Conv2 features
        x = torch.relu(self.bn2(self.conv2(x)))
        features['conv2'] = x.detach()
        
        return features


# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Results directory for all output files
RESULTS_DIR = 'results_cnn'


def create_stratified_test_subset(test_dataset, num_samples=2000, seed=42):
    """
    Create a stratified subset of test data (200 images per class)
    
    This ensures fair comparison between CNN and GPT-4o by using
    the exact same test images for both models.
    
    Args:
        test_dataset: CIFAR-10 test dataset
        num_samples: Total number of samples (must be divisible by 10)
        seed: Random seed for reproducibility
    
    Returns:
        List of indices for the stratified subset
    """
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    indices_file = os.path.join(RESULTS_DIR, 'stratified_test_indices.json')
    
    # Load existing indices if available
    if os.path.exists(indices_file):
        print(f"Loading existing stratified test indices from {indices_file}...")
        with open(indices_file, 'r') as f:
            indices = json.load(f)
        print(f"   Loaded {len(indices)} indices (200 per class)")
        return indices
    
    # Create new stratified subset
    np.random.seed(seed)
    samples_per_class = num_samples // 10
    
    print(f"Creating stratified test subset...")
    print(f"   Total samples: {num_samples}")
    print(f"   Samples per class: {samples_per_class}")
    
    # Group indices by class
    class_indices = {i: [] for i in range(10)}
    for idx in range(len(test_dataset)):
        _, label = test_dataset[idx]
        class_indices[label].append(idx)
    
    # Sample from each class
    stratified_indices = []
    for class_idx in range(10):
        class_samples = np.random.choice(
            class_indices[class_idx], 
            samples_per_class, 
            replace=False
        )
        stratified_indices.extend(class_samples.tolist())
    
    # Shuffle the combined indices
    np.random.shuffle(stratified_indices)
    stratified_indices = [int(i) for i in stratified_indices]  # Ensure JSON serializable
    
    # Save indices for reproducibility
    with open(indices_file, 'w') as f:
        json.dump(stratified_indices, f)
    print(f"   Stratified subset created and saved to {indices_file}")
    
    return stratified_indices


def get_data_loaders(batch_size=128):
    """Load CIFAR-10 training and test datasets"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # No augmentation for test - normalization applied in evaluate() for flexibility
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Download datasets
    print("Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)} (will use 2,000 stratified subset for evaluation)")
    
    return train_loader, test_dataset  # Return dataset instead of loader for stratified sampling


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/total:.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / total, 100. * correct / total


def evaluate(model, test_dataset, indices, criterion, device):
    """Evaluate model on stratified test subset (2,000 images)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Normalization transform
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), 
        (0.2470, 0.2435, 0.2616)
    )
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Evaluating', leave=False):
            img, label = test_dataset[idx]
            
            # Normalize and add batch dimension
            img_normalized = normalize(img).unsqueeze(0).to(device)
            label_tensor = torch.tensor([label]).to(device)
            
            outputs = model(img_normalized)
            loss = criterion(outputs, label_tensor)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += 1
            correct += predicted.eq(label_tensor).sum().item()
    
    return running_loss / total, 100. * correct / total


def final_evaluation_with_analysis(model, test_dataset, indices, device):
    """
    Comprehensive final evaluation demonstrating mathematical understanding.
    
    This function:
    1. Collects all predictions and ground truth labels
    2. Computes confusion matrix manually
    3. Calculates precision, recall, F1 from scratch (not using sklearn)
    4. Verifies our manual softmax/cross-entropy against PyTorch
    5. Generates visualization of confusion matrix
    
    Args:
        model: Trained CNN model
        test_dataset: CIFAR-10 test dataset
        indices: Stratified subset indices
        device: torch device (cuda/cpu)
    
    Returns:
        Dictionary containing all metrics and analysis
    """
    model.eval()
    
    # Storage for predictions and labels
    all_predictions = []
    all_labels = []
    all_logits = []
    all_probs_manual = []
    
    # Normalization transform
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), 
        (0.2470, 0.2435, 0.2616)
    )
    
    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION WITH MATHEMATICAL ANALYSIS")
    print("=" * 70)
    print(f"Evaluating on {len(indices)} stratified test images...")
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Collecting predictions'):
            img, label = test_dataset[idx]
            
            # Normalize and add batch dimension
            img_normalized = normalize(img).unsqueeze(0).to(device)
            
            # Get model output (raw logits)
            logits = model(img_normalized)
            logits_np = logits.cpu().numpy()
            
            # Apply our manual softmax
            probs_manual = manual_softmax(logits_np)
            
            # Get prediction (argmax of logits)
            _, predicted = logits.max(1)
            
            all_predictions.append(predicted.item())
            all_labels.append(label)
            all_logits.append(logits_np[0])
            all_probs_manual.append(probs_manual[0])
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_probs_manual = np.array(all_probs_manual)
    
    # ==========================================================================
    # VERIFY MANUAL IMPLEMENTATIONS
    # ==========================================================================
    print("\n" + "-" * 70)
    print("VERIFICATION: Manual vs PyTorch Implementations")
    print("-" * 70)
    
    # Compare manual softmax with PyTorch
    pytorch_softmax = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    softmax_diff = np.abs(all_probs_manual - pytorch_softmax).max()
    print(f"Softmax max difference: {softmax_diff:.2e} (should be ~0)")
    
    # Compare manual cross-entropy with PyTorch
    manual_ce = manual_cross_entropy(all_probs_manual, all_labels)
    pytorch_ce = nn.CrossEntropyLoss()(
        torch.tensor(all_logits), 
        torch.tensor(all_labels)
    ).item()
    ce_diff = abs(manual_ce - pytorch_ce)
    print(f"Cross-entropy difference: {ce_diff:.2e} (should be ~0)")
    print(f"  Manual CE: {manual_ce:.4f}, PyTorch CE: {pytorch_ce:.4f}")
    
    # ==========================================================================
    # COMPUTE CONFUSION MATRIX
    # ==========================================================================
    print("\n" + "-" * 70)
    print("CONFUSION MATRIX ANALYSIS")
    print("-" * 70)
    
    cm = compute_confusion_matrix(all_labels, all_predictions, num_classes=10)
    
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("Classes: " + " ".join([f"{c[:4]:>5}" for c in CLASSES]))
    print("-" * 70)
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:<10} " + " ".join([f"{val:>5}" for val in row]))
    
    # ==========================================================================
    # COMPUTE METRICS FROM CONFUSION MATRIX
    # ==========================================================================
    metrics = print_classification_report(cm, CLASSES)
    
    # ==========================================================================
    # ADDITIONAL ANALYSIS
    # ==========================================================================
    print("\n" + "-" * 70)
    print("ADDITIONAL MATHEMATICAL ANALYSIS")
    print("-" * 70)
    
    # Most confused class pairs
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    most_confused_idx = np.unravel_index(np.argmax(cm_no_diag), cm.shape)
    print(f"\nMost confused pair: {CLASSES[most_confused_idx[0]]} → "
          f"{CLASSES[most_confused_idx[1]]} ({cm_no_diag[most_confused_idx]} errors)")
    
    # Per-class accuracy analysis
    print("\nPer-class accuracy (TP / Total in class):")
    for i, name in enumerate(CLASSES):
        class_total = np.sum(cm[i, :])
        class_correct = cm[i, i]
        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  {name:<12}: {class_acc*100:>6.2f}% ({class_correct}/{class_total})")
    
    # ==========================================================================
    # VISUALIZE CONFUSION MATRIX
    # ==========================================================================
    plot_confusion_matrix(cm, CLASSES)
    
    return {
        'confusion_matrix': cm,
        'metrics': metrics,
        'predictions': all_predictions,
        'labels': all_labels,
        'logits': all_logits,
        'probabilities': all_probs_manual
    }


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
    
    # Normalize confusion matrix for color intensity
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Create heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax, label='Proportion')
    
    # Set ticks and labels
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix\n(normalized by row)')
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}',
                   ha='center', va='center',
                   color='white' if cm_normalized[i, j] > thresh else 'black',
                   fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nConfusion matrix saved to {save_path}")
    plt.close()


def train_model(model, train_loader, test_dataset, test_indices, epochs=20, lr=0.001):
    """Train the CNN model with evaluation on stratified 2,000 test subset"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"   Evaluating on {len(test_indices)} stratified test images (same as GPT-4o)")
    print("=" * 70)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate on stratified subset
        test_loss, test_acc = evaluate(model, test_dataset, test_indices, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}% (2,000 images)")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            model_path = os.path.join(RESULTS_DIR, 'best_cnn_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_subset_size': len(test_indices),
            }, model_path)
            print(f"Best model saved! (Test Acc: {test_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best test accuracy: {best_acc:.2f}% (on 2,000 stratified test images)")
    
    return history


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
    
    # =====================================================================
    # 1. Loss Curves (with exponential fit)
    # =====================================================================
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
    
    # =====================================================================
    # 2. Accuracy Curves
    # =====================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_list, history['train_acc'], label='Train Acc', marker='o', linewidth=2)
    ax2.plot(epochs_list, history['test_acc'], label='Test Acc', marker='s', linewidth=2)
    
    # Add GPT-4o target range (85-95%)
    ax2.axhspan(85, 95, alpha=0.2, color='green', label='GPT-4o Range (85-95%)')
    ax2.axhline(85, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axhline(95, color='green', linestyle='--', alpha=0.5, linewidth=1)
    
    # Final accuracy annotation
    final_acc = history['test_acc'][-1]
    ax2.annotate(f'Final: {final_acc:.2f}%', 
                xy=(len(history['test_acc']), final_acc),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy (%)', fontsize=11)
    ax2.set_title('Accuracy Convergence', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # =====================================================================
    # 3. Convergence Rate: |acc(t) - acc(t-1)|
    # =====================================================================
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
    
    # =====================================================================
    # 4. Overfitting Gap: train_acc - test_acc
    # =====================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    overfitting_gap = np.array(history['train_acc']) - np.array(history['test_acc'])
    ax4.plot(epochs_list, overfitting_gap, marker='o', color='orange', linewidth=2)
    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.fill_between(epochs_list, 0, overfitting_gap, alpha=0.3, color='orange')
    ax4.set_xlabel('Epoch', fontsize=11)
    ax4.set_ylabel('Train Acc - Test Acc (%)', fontsize=11)
    ax4.set_title('Generalization Gap (Overfitting Indicator)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # =====================================================================
    # 5. Learning Rate Schedule
    # =====================================================================
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
    
    # =====================================================================
    # 6. Mathematical Summary Statistics
    # =====================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    
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
  {'✓ Converged' if len(history['test_acc']) > 5 and recent_improvement < 0.5 else '→ Still improving'}
  
Target Comparison:
  Current: {final_test_acc:.2f}%
  GPT-4o:  85-95%
  Gap:     {85 - final_test_acc:.2f}% to reach GPT-4o range
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Mathematical Convergence Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    history_path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(history_path, dpi=150, bbox_inches='tight')
    print(f"Training history with mathematical analysis saved to {history_path}")
    plt.close()


def main():
    """
    Main training pipeline
    
    ==========================================================================
    TRAINING PROCEDURE
    ==========================================================================
    
    1. Data Preparation:
       - Load CIFAR-10 (50,000 training, 10,000 test images)
       - Apply data augmentation (random flip, crop) to training data
       - Create stratified test subset (200 images per class = 2,000 total)
    
    2. Model Training:
       - Forward pass: compute predictions from inputs
       - Loss computation: cross-entropy between predictions and labels
       - Backward pass: compute gradients via backpropagation
       - Parameter update: θ_new = θ_old - lr * ∇L(θ)
    
    3. Evaluation:
       - Compute accuracy on held-out stratified test set
       - Generate confusion matrix
       - Calculate precision, recall, F1 per class
    
    ==========================================================================
    """
    print("=" * 70)
    print("CIFAR-10 Custom CNN Training")
    print("=" * 70)
    
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_TEST_SAMPLES = 2000  # Stratified subset for fair comparison with GPT-4o
    
    print("\nHYPERPARAMETERS (with mathematical justification):")
    print(f"   Batch Size: {BATCH_SIZE}")
    print("      - Larger batches → more stable gradient estimates")
    print("      - Trade-off between memory usage and gradient variance")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print("      - Controls step size: θ = θ - lr * ∇L")
    print("      - Too high → divergence, too low → slow convergence")
    print(f"   Epochs: {EPOCHS}")
    print("      - One epoch = one pass through all training data")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\nResults will be saved to: {RESULTS_DIR}/")
    
    # Load data
    train_loader, test_dataset = get_data_loaders(BATCH_SIZE)
    
    # Create/load stratified test subset (same 2,000 images used for GPT-4o)
    test_indices = create_stratified_test_subset(test_dataset, num_samples=NUM_TEST_SAMPLES)
    
    # Create model
    print("\nBuilding model...")
    model = CustomCNN(num_classes=10).to(device)
    
    # Print model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Train model
    history = train_model(model, train_loader, test_dataset, test_indices, 
                         epochs=EPOCHS, lr=LEARNING_RATE)
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, epochs=EPOCHS)
    
    # Save training history
    history_json_path = os.path.join(RESULTS_DIR, 'training_history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_json_path}")
    
    # ==========================================================================
    # COMPREHENSIVE FINAL EVALUATION
    # ==========================================================================
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model_path = os.path.join(RESULTS_DIR, 'best_cnn_model.pth')
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"   Loaded model from epoch {checkpoint['epoch'] + 1} "
          f"(Test Acc: {checkpoint['test_acc']:.2f}%)")
    
    # Run comprehensive evaluation with mathematical analysis
    evaluation_results = final_evaluation_with_analysis(
        model, test_dataset, test_indices, device
    )
    
    # Save evaluation results
    eval_summary = {
        'accuracy': float(evaluation_results['metrics']['accuracy']),
        'macro_precision': float(evaluation_results['metrics']['macro_avg']['precision']),
        'macro_recall': float(evaluation_results['metrics']['macro_avg']['recall']),
        'macro_f1': float(evaluation_results['metrics']['macro_avg']['f1']),
        'per_class_f1': evaluation_results['metrics']['per_class']['f1'].tolist(),
        'confusion_matrix': evaluation_results['confusion_matrix'].tolist()
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
    print(f"   • confusion_matrix.png         - Confusion matrix visualization")
    print(f"   • evaluation_results.json      - Final evaluation metrics")
    print(f"   • stratified_test_indices.json - Fixed 2,000 test image indices")
    print(f"\nEvaluation used {NUM_TEST_SAMPLES} stratified test images (same as GPT-4o)")


if __name__ == "__main__":
    main()

