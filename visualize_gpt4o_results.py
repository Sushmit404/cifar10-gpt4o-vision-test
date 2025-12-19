"""
Generate visualizations from GPT-4o evaluation results (32x32 non-upscaled).
Creates confusion matrix and per-class performance charts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

def load_gpt4o_results():
    """Load the most recent GPT-4o 32x32 results file"""
    results_files = glob.glob('results_gpt4o_32x32/gpt4o_32x32_results_*.json')
    if not results_files:
        raise FileNotFoundError("No GPT-4o 32x32 results found in results_gpt4o_32x32/")
    
    # Get most recent
    latest_file = max(results_files, key=os.path.getmtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(cm, class_names, save_path):
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
           title='GPT-4o Confusion Matrix')
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


def plot_per_class_performance(results, save_path):
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
    ax.set_title('GPT-4o Per-Class Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=45, ha='right')
    
    # Set y-axis limits based on accuracy range
    max_val = max(max(precision), max(recall), max(accuracy))
    min_val = min(min(precision), min(recall), min(accuracy))
    y_range = max_val - min_val
    ax.set_ylim([max(85, min_val - y_range * 0.1), min(102, max_val + y_range * 0.1)])
    ax.grid(axis='y', alpha=0.3)
    
    # Add overall accuracy line (same as in class accuracy chart)
    ax.axhline(y=results['accuracy'] * 100, color='blue', linestyle='--', 
               linewidth=2, alpha=0.7, label=f"Overall Accuracy: {results['accuracy']*100:.1f}%")
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_class_accuracy_comparison(results, save_path):
    """Plot per-class accuracy as bar chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    accuracies = [results['per_class_metrics'][c]['recall'] * 100 for c in CLASSES]
    colors = plt.cm.RdYlGn(np.array(accuracies) / 100)
    
    bars = ax.bar(CLASSES, accuracies, color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title(f'GPT-4o Per-Class Accuracy - Overall: {results["accuracy"]*100:.2f}%', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([90, 102])
    ax.axhline(y=results['accuracy'] * 100, color='blue', linestyle='--', 
               linewidth=2, alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_summary_dashboard(results, save_path):
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
    ax2.set_ylim([90, 102])
    ax2.axhline(y=results['accuracy'] * 100, color='blue', linestyle='--', linewidth=2)
    ax2.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
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
    ax3.set_ylim([max(88, min_val - y_range * 0.1), min(102, max_val + y_range * 0.1)])
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
    
    # Find best and worst classes dynamically
    sorted_classes = sorted(CLASSES, key=lambda c: results['per_class_metrics'][c]['recall'], reverse=True)
    best_classes = sorted_classes[:3]
    worst_classes = sorted_classes[-3:]
    
    summary = f"""
GPT-4o CIFAR-10 EVALUATION SUMMARY
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
    
    plt.suptitle('GPT-4o Vision - CIFAR-10 Evaluation Results', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("GPT-4o 32×32 Results Visualization")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'gpt_4o_visualizations_32x32'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}/")
    
    # Load results
    results = load_gpt4o_results()
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Total Images: {results['total_valid']}")
    
    # Generate visualizations in output folder
    print("\nGenerating visualizations...")
    
    plot_confusion_matrix(
        results['confusion_matrix'], 
        CLASSES, 
        os.path.join(output_dir, '2gpt4o_32x32_confusion_matrix.png')
    )
    
    plot_per_class_performance(
        results, 
        os.path.join(output_dir, '2gpt4o_32x32_per_class_performance.png')
    )
    
    plot_class_accuracy_comparison(
        results, 
        os.path.join(output_dir, '2gpt4o_32x32_class_accuracy.png')
    )
    
    plot_summary_dashboard(
        results, 
        os.path.join(output_dir, '2gpt4o_32x32_summary_dashboard.png')
    )
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files (in {output_dir}/):")
    print("  - 2gpt4o_32x32_confusion_matrix.png      - Confusion matrix heatmap")
    print("  - 2gpt4o_32x32_per_class_performance.png - Precision/Recall/Accuracy by class")
    print("  - 2gpt4o_32x32_class_accuracy.png        - Per-class accuracy bars")
    print("  - 2gpt4o_32x32_summary_dashboard.png     - Combined dashboard")


if __name__ == "__main__":
    main()

