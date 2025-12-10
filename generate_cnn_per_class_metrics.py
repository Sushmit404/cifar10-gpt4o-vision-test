"""
Generate per-class accuracy and metrics from saved CNN models.
This script loads saved models and evaluates them to compute detailed metrics
without retraining. Optionally generates visualizations.

Usage:
    python generate_cnn_per_class_metrics.py --results-dir results_cnn
    python generate_cnn_per_class_metrics.py --results-dir results_cnn_32_20251209_132041 --visualize
    python generate_cnn_per_class_metrics.py --results-dir results_cnn --visualize --output-dir cnn_visualizations
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


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


class SimpleCNN(nn.Module):
    """Old simple 2-layer CNN architecture"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CustomCNN(nn.Module):
    """New ResNet-style CNN architecture"""
    def __init__(self, num_classes=10, input_size=32):
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
        
        # FC layer size depends on input size
        if input_size == 32:
            self.fc1 = nn.Linear(256 * 2 * 2, 256)
        else:  # 224
            self.fc1 = nn.Linear(256 * 14 * 14, 256)
        
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Weight initialization
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


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
    """Compute confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm


def compute_metrics_from_confusion_matrix(cm, class_names=None):
    """Compute precision, recall, and F1-score from confusion matrix"""
    num_classes = cm.shape[0]
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support[i] = np.sum(cm[i, :])
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0.0
    
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
        'accuracy': np.trace(cm) / np.sum(cm)
    }
    
    return metrics


def load_stratified_indices():
    """Load stratified test subset indices"""
    indices_file = 'stratified_subset_2000.json'
    if os.path.exists(indices_file):
        with open(indices_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'indices' in data:
                return data['indices']
            return data
    raise FileNotFoundError(f"Stratified subset file not found: {indices_file}")


def detect_model_architecture(checkpoint):
    """Detect model architecture from checkpoint keys"""
    state_dict_keys = set(checkpoint.get('model_state_dict', checkpoint).keys())
    
    # Check for old simple architecture
    if 'conv1.weight' in state_dict_keys and 'initial_conv' not in state_dict_keys:
        return 'simple'
    # Otherwise it's the new ResNet architecture
    return 'resnet'


def detect_input_size(results_dir, architecture='resnet'):
    """Detect input size from results directory name or existing evaluation"""
    # Check if it's a 224 model
    if '224' in results_dir or 'results_cnn_224' in results_dir:
        return 224
    # Default to 32
    return 32


def evaluate_model(model, test_dataset, indices, device, input_size=32):
    """Evaluate model on stratified test subset"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    
    # Upscale transform for 224 models
    if input_size == 224:
        from PIL import Image
        upscale_transform = transforms.Lambda(lambda x: x.resize((224, 224), Image.BILINEAR))
    else:
        upscale_transform = None
    
    print(f"\nEvaluating on {len(indices)} test images...")
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='Evaluating'):
            img, label = test_dataset[idx]
            
            # Upscale if needed
            if upscale_transform:
                img = upscale_transform(img)
            
            img_normalized = normalize(img).unsqueeze(0).to(device)
            
            outputs = model(img_normalized)
            _, predicted = outputs.max(1)
            
            all_predictions.append(predicted.item())
            all_labels.append(label)
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    cm = compute_confusion_matrix(all_labels, all_predictions)
    metrics = compute_metrics_from_confusion_matrix(cm, CLASSES)
    
    return {'confusion_matrix': cm.tolist(), 'metrics': metrics, 
            'predictions': all_predictions.tolist(), 'labels': all_labels.tolist()}


def format_per_class_metrics(metrics, class_names):
    """Format per-class metrics as dictionary"""
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        per_class_metrics[class_name] = {
            'precision': float(metrics['per_class']['precision'][i]),
            'recall': float(metrics['per_class']['recall'][i]),
            'f1': float(metrics['per_class']['f1'][i]),
            'correct': int(metrics['per_class']['support'][i] * metrics['per_class']['recall'][i]),
            'total': int(metrics['per_class']['support'][i])
        }
    return per_class_metrics


def plot_confusion_matrix(cm, class_names, save_path, model_name="CNN"):
    """Plot confusion matrix as heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    cm = np.array(cm)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(len(class_names)), 
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, 
           yticklabels=class_names,
           ylabel='True Label', 
           xlabel='Predicted Label', 
           title=f'{model_name} Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    thresh = cm_norm.max() / 2.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                   color='white' if cm_norm[i, j] > thresh else 'black', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_per_class_performance(results, save_path, model_name="CNN"):
    """Plot per-class precision, recall, and accuracy as grouped bar chart with overall accuracy line"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(CLASSES))
    width = 0.25
    
    precision = [results['per_class_metrics'][c]['precision'] * 100 for c in CLASSES]
    recall = [results['per_class_metrics'][c]['recall'] * 100 for c in CLASSES]
    accuracy = [results['per_class_metrics'][c]['recall'] * 100 for c in CLASSES]  # Per-class accuracy is recall
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72')
    bars3 = ax.bar(x + width, accuracy, width, label='Accuracy', color='#F18F01')
    
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title(f'{model_name} Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    
    # Set y-axis limits based on accuracy range
    max_val = max(max(precision), max(recall), max(accuracy))
    min_val = min(min(precision), min(recall), min(accuracy))
    y_range = max_val - min_val
    ax.set_ylim([max(0, min_val - y_range * 0.1), min(100, max_val + y_range * 0.1)])
    ax.grid(axis='y', alpha=0.3)
    
    # Add overall accuracy line (same as in class accuracy chart)
    ax.axhline(y=results['accuracy'] * 100, color='blue', linestyle='--', 
               linewidth=2, alpha=0.7, label=f"Overall Accuracy: {results['accuracy']*100:.1f}%")
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_class_accuracy_comparison(results, save_path, model_name="CNN"):
    """Plot per-class accuracy as bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    accuracies = [results['per_class_metrics'][c]['recall'] * 100 for c in CLASSES]
    colors = plt.cm.RdYlGn(np.array(accuracies) / 100)
    
    bars = ax.bar(CLASSES, accuracies, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title(f'{model_name} Per-Class Accuracy (Overall: {results["accuracy"]*100:.2f}%)', 
                 fontsize=14, fontweight='bold')
    
    # Set y-axis limits based on accuracy range
    max_acc = max(accuracies)
    min_acc = min(accuracies)
    y_range = max_acc - min_acc
    ax.set_ylim([max(0, min_acc - y_range * 0.1), min(100, max_acc + y_range * 0.1)])
    
    ax.axhline(y=results['accuracy'] * 100, color='blue', linestyle='--', 
               linewidth=2, alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_summary_dashboard(results, save_path, model_name="CNN", input_size=32):
    """Create a summary dashboard with multiple visualizations"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    cm = np.array(results['confusion_matrix'])
    
    # Confusion matrix
    ax1 = fig.add_subplot(gs[0, 0])
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im = ax1.imshow(cm_norm, cmap='Blues')
    fig.colorbar(im, ax=ax1, fraction=0.046)
    ax1.set_xticks(np.arange(len(CLASSES)))
    ax1.set_yticks(np.arange(len(CLASSES)))
    ax1.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=8)
    ax1.set_yticklabels(CLASSES, fontsize=8)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    thresh = cm_norm.max() / 2.
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black', fontsize=7)
    
    # Per-class accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    accuracies = [results['per_class_metrics'][c]['recall'] * 100 for c in CLASSES]
    colors = plt.cm.RdYlGn(np.array(accuracies) / 100)
    bars = ax2.bar(range(len(CLASSES)), accuracies, color=colors, edgecolor='black')
    ax2.set_xticks(range(len(CLASSES)))
    ax2.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
    max_acc = max(accuracies)
    min_acc = min(accuracies)
    y_range = max_acc - min_acc
    ax2.set_ylim([max(0, min_acc - y_range * 0.1), min(100, max_acc + y_range * 0.1)])
    ax2.axhline(y=results['accuracy'] * 100, color='blue', linestyle='--', linewidth=2)
    ax2.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # Precision/Recall/Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(CLASSES))
    width = 0.25
    precision = [results['per_class_metrics'][c]['precision'] * 100 for c in CLASSES]
    recall = [results['per_class_metrics'][c]['recall'] * 100 for c in CLASSES]
    accuracy = [results['per_class_metrics'][c]['recall'] * 100 for c in CLASSES]  # Per-class accuracy is recall
    
    ax3.bar(x - width, precision, width, label='Precision', color='#2E86AB')
    ax3.bar(x, recall, width, label='Recall', color='#A23B72')
    ax3.bar(x + width, accuracy, width, label='Accuracy', color='#F18F01')
    ax3.set_xticks(x)
    ax3.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Score (%)')
    ax3.set_title('Precision / Recall / Accuracy', fontsize=12, fontweight='bold')
    max_val = max(max(precision), max(recall), max(accuracy))
    min_val = min(min(precision), min(recall), min(accuracy))
    y_range = max_val - min_val
    ax3.set_ylim([max(0, min_val - y_range * 0.1), min(100, max_val + y_range * 0.1)])
    # Add overall accuracy line
    ax3.axhline(y=results['accuracy'] * 100, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(axis='y', alpha=0.3)
    
    # Summary statistics
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Find most confused pairs
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    top_confusions = []
    for _ in range(5):
        idx = np.unravel_index(np.argmax(cm_no_diag), cm.shape)
        if cm_no_diag[idx] > 0:
            top_confusions.append((CLASSES[idx[0]], CLASSES[idx[1]], cm_no_diag[idx]))
            cm_no_diag[idx] = 0
    
    confusion_text = "\n".join([f"  {t} -> {p}: {c} errors" for t, p, c in top_confusions])
    
    # Find best and worst classes
    sorted_classes = sorted(CLASSES, key=lambda c: results['per_class_metrics'][c]['recall'], reverse=True)
    best_classes = sorted_classes[:3]
    worst_classes = sorted_classes[-3:]
    
    summary = f"""
{model_name} CIFAR-10 EVALUATION SUMMARY
Input Size: {input_size}×{input_size}
{'='*45}

Overall Metrics:
  Accuracy:        {results['accuracy']*100:.2f}%
  Correct:         {results['correct']} / {results['total_valid']}
  
  Macro Precision: {results['macro_precision']*100:.2f}%
  Macro Recall:    {results['macro_recall']*100:.2f}%
  Macro F1:        {results['macro_f1']*100:.2f}%

Best Performing Classes:
  {best_classes[0]}:     {results['per_class_metrics'][best_classes[0]]['recall']*100:.1f}%
  {best_classes[1]}: {results['per_class_metrics'][best_classes[1]]['recall']*100:.1f}%
  {best_classes[2]}:    {results['per_class_metrics'][best_classes[2]]['recall']*100:.1f}%

Most Challenging Classes:
  {worst_classes[0]}:      {results['per_class_metrics'][worst_classes[0]]['recall']*100:.1f}%
  {worst_classes[1]}:      {results['per_class_metrics'][worst_classes[1]]['recall']*100:.1f}%
  {worst_classes[2]}:     {results['per_class_metrics'][worst_classes[2]]['recall']*100:.1f}%

Top Confusion Pairs:
{confusion_text}
"""
    
    ax4.text(0.05, 0.95, summary, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle(f'{model_name} - CIFAR-10 Evaluation Results ({input_size}×{input_size})', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate per-class metrics from saved CNN models')
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Results directory containing best_cnn_model.pth')
    parser.add_argument('--input-size', type=int, default=None,
                        help='Input size (32 or 224). Auto-detected if not specified')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for visualizations (default: results_dir)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    model_path = os.path.join(results_dir, 'best_cnn_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load existing evaluation results if available
    eval_results_path = os.path.join(results_dir, 'evaluation_results.json')
    existing_results = {}
    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            existing_results = json.load(f)
        print(f"Loaded existing evaluation results from {eval_results_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint to detect architecture
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, weights_only=True)
    architecture = detect_model_architecture(checkpoint)
    print(f"Detected architecture: {architecture}")
    
    # Detect input size
    input_size = args.input_size if args.input_size else detect_input_size(results_dir, architecture)
    print(f"Detected input size: {input_size}×{input_size}")
    
    # Create appropriate model
    if architecture == 'simple':
        model = SimpleCNN().to(device)
    else:
        model = CustomCNN(input_size=input_size).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded model (epoch {checkpoint.get('epoch', 'unknown') + 1}, "
          f"test_acc: {checkpoint.get('test_acc', 0):.2f}%)")
    
    # Load test data
    print("\nLoading CIFAR-10 test dataset...")
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Load stratified indices
    test_indices = load_stratified_indices()
    print(f"Loaded {len(test_indices)} stratified test indices")
    
    # Evaluate model
    results = evaluate_model(model, test_dataset, test_indices, device, input_size)
    
    # Format metrics
    per_class_metrics = format_per_class_metrics(results['metrics'], CLASSES)
    
    # Create enhanced results
    enhanced_results = {
        'accuracy': float(results['metrics']['accuracy']),
        'macro_precision': float(results['metrics']['macro_avg']['precision']),
        'macro_recall': float(results['metrics']['macro_avg']['recall']),
        'macro_f1': float(results['metrics']['macro_avg']['f1']),
        'weighted_precision': float(results['metrics']['weighted_avg']['precision']),
        'weighted_recall': float(results['metrics']['weighted_avg']['recall']),
        'weighted_f1': float(results['metrics']['weighted_avg']['f1']),
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': results['confusion_matrix'],
        'correct': int(results['metrics']['accuracy'] * len(test_indices)),
        'total_valid': len(test_indices),
        'input_size': input_size,
        'epochs': existing_results.get('epochs', 'unknown'),
        'early_stopped': existing_results.get('early_stopped', False),
        'timing': existing_results.get('timing', {})
    }
    
    # Save enhanced results
    enhanced_results_path = os.path.join(results_dir, 'evaluation_results_enhanced.json')
    with open(enhanced_results_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f"\n✅ Enhanced evaluation results saved to {enhanced_results_path}")
    
    # Generate visualizations if requested
    if args.visualize:
        output_dir = args.output_dir if args.output_dir else results_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine model name from directory
        model_name = os.path.basename(results_dir.rstrip('/\\'))
        if '32' in model_name or input_size == 32:
            model_name = f"CNN ({input_size}×{input_size})"
        else:
            model_name = f"CNN ({input_size}×{input_size})"
        
        print(f"\n{'='*70}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*70}")
        print(f"Output directory: {output_dir}/")
        
        plot_confusion_matrix(
            enhanced_results['confusion_matrix'],
            CLASSES,
            os.path.join(output_dir, '2cnn_confusion_matrix.png'),
            model_name
        )
        
        plot_per_class_performance(
            enhanced_results,
            os.path.join(output_dir, '2cnn_per_class_performance.png'),
            model_name
        )
        
        plot_class_accuracy_comparison(
            enhanced_results,
            os.path.join(output_dir, '2cnn_class_accuracy.png'),
            model_name
        )
        
        plot_summary_dashboard(
            enhanced_results,
            os.path.join(output_dir, '2cnn_summary_dashboard.png'),
            model_name,
            input_size
        )
        
        print(f"\n✅ Visualizations saved to {output_dir}/")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PER-CLASS METRICS SUMMARY")
    print("=" * 70)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Correct':>8} {'Total':>8}")
    print("-" * 70)
    
    for class_name in CLASSES:
        metrics = per_class_metrics[class_name]
        print(f"{class_name:<12} {metrics['precision']*100:>9.2f}% "
              f"{metrics['recall']*100:>9.2f}% {metrics['f1']*100:>9.2f}% "
              f"{metrics['correct']:>8} {metrics['total']:>8}")
    
    print("-" * 70)
    print(f"{'Overall':<12} {enhanced_results['macro_precision']*100:>9.2f}% "
          f"{enhanced_results['macro_recall']*100:>9.2f}% "
          f"{enhanced_results['macro_f1']*100:>9.2f}% "
          f"{enhanced_results['correct']:>8} {enhanced_results['total_valid']:>8}")
    print(f"\nOverall Accuracy: {enhanced_results['accuracy']*100:.2f}%")


if __name__ == "__main__":
    main()

