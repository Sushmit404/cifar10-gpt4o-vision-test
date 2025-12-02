"""
CIFAR-10: Trained CNN vs. GPT-4o Vision Zero-Shot Comparison
Part 2: Evaluate and compare both models on stratified test subset
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import io
import base64
from openai import OpenAI
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv
import time

# Import CNN model
from train_cnn import CustomCNN

# Load environment variables
load_dotenv()

# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_stratified_test_subset(test_dataset, num_samples=2000, seed=42):
    """
    Create a stratified subset of test data (200 images per class)
    
    Args:
        test_dataset: CIFAR-10 test dataset
        num_samples: Total number of samples (must be divisible by 10)
        seed: Random seed for reproducibility
    
    Returns:
        List of indices for the stratified subset
    """
    np.random.seed(seed)
    samples_per_class = num_samples // 10
    
    print(f"📊 Creating stratified test subset...")
    print(f"   Total samples: {num_samples}")
    print(f"   Samples per class: {samples_per_class}")
    
    # Group indices by class
    class_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_dataset):
        class_indices[label].append(idx)
    
    # Sample from each class
    stratified_indices = []
    for class_idx in range(10):
        class_samples = np.random.choice(
            class_indices[class_idx], 
            samples_per_class, 
            replace=False
        )
        stratified_indices.extend(class_samples)
    
    # Shuffle the combined indices
    np.random.shuffle(stratified_indices)
    
    print(f"✅ Stratified subset created with {len(stratified_indices)} samples")
    
    # Save indices for reproducibility
    with open('stratified_test_indices.json', 'w') as f:
        json.dump(stratified_indices, f)
    print("💾 Indices saved to stratified_test_indices.json")
    
    return stratified_indices


def load_trained_cnn(model_path='best_cnn_model.pth'):
    """Load the trained CNN model"""
    print(f"\n🏗️  Loading trained CNN from {model_path}...")
    
    model = CustomCNN(num_classes=10).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (Test Acc from training: {checkpoint['test_acc']:.2f}%)")
    
    return model


def evaluate_cnn(model, test_dataset, indices):
    """Evaluate CNN on test subset"""
    print("\n🔍 Evaluating CNN on test subset...")
    
    model.eval()
    predictions = []
    true_labels = []
    
    # Normalization for CNN (must match training)
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), 
        (0.2470, 0.2435, 0.2616)
    )
    
    with torch.no_grad():
        for idx in tqdm(indices, desc='CNN Evaluation'):
            img, label = test_dataset[idx]
            
            # Get raw image and normalize
            img_tensor = normalize(img).unsqueeze(0).to(device)
            
            # Predict
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            
            predictions.append(pred)
            true_labels.append(label)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    accuracy = (predictions == true_labels).mean() * 100
    print(f"✅ CNN Accuracy: {accuracy:.2f}%")
    
    return predictions, true_labels


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    img = tensor.numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def classify_with_gpt4o(image_pil, api_key):
    """
    Classify image using GPT-4o Vision
    
    Args:
        image_pil: PIL Image
        api_key: OpenAI API key
    
    Returns:
        Predicted class name (lowercase)
    """
    # Upscale to 224x224 for better vision performance
    img_upscaled = image_pil.resize((224, 224), Image.BILINEAR)
    
    # Convert to bytes
    buffer = io.BytesIO()
    img_upscaled.save(buffer, format='PNG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Call API
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Classify this image as exactly one of: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Return only the label, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=10
    )
    
    return response.choices[0].message.content.strip().lower()


def evaluate_gpt4o(test_dataset, indices, api_key, save_progress=True):
    """Evaluate GPT-4o Vision on test subset"""
    print("\n🔍 Evaluating GPT-4o Vision on test subset...")
    print("⚠️  This will make 2000 API calls and may take ~30-60 minutes")
    print("⚠️  Estimated cost: ~$10-20 (depending on pricing)")
    
    response = input("\nProceed with GPT-4o evaluation? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ GPT-4o evaluation skipped.")
        return None, None
    
    predictions = []
    true_labels = []
    failed_indices = []
    
    # Load progress if exists
    progress_file = 'gpt4o_progress.json'
    if os.path.exists(progress_file) and save_progress:
        print(f"📂 Found existing progress file: {progress_file}")
        with open(progress_file, 'r') as f:
            progress = json.load(f)
            predictions = progress['predictions']
            true_labels = progress['true_labels']
            failed_indices = progress.get('failed_indices', [])
            start_idx = len(predictions)
        print(f"   Resuming from index {start_idx}/{len(indices)}")
    else:
        start_idx = 0
    
    # Process images
    for i, idx in enumerate(tqdm(indices[start_idx:], desc='GPT-4o Evaluation', initial=start_idx, total=len(indices))):
        img, label = test_dataset[idx]
        img_pil = tensor_to_pil(img)
        
        try:
            pred_name = classify_with_gpt4o(img_pil, api_key)
            
            # Map to class index
            if pred_name in CLASSES:
                pred_idx = CLASSES.index(pred_name)
            else:
                # Handle unexpected responses
                print(f"\n⚠️  Unexpected response: '{pred_name}' for index {idx}")
                pred_idx = -1  # Mark as error
                failed_indices.append(idx)
            
            predictions.append(pred_idx)
            true_labels.append(label)
            
            # Save progress every 50 images
            if save_progress and (len(predictions) % 50 == 0):
                with open(progress_file, 'w') as f:
                    json.dump({
                        'predictions': predictions,
                        'true_labels': true_labels,
                        'failed_indices': failed_indices
                    }, f)
            
            # Rate limiting (to avoid hitting API limits)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\n❌ Error at index {idx}: {str(e)}")
            predictions.append(-1)  # Mark as error
            true_labels.append(label)
            failed_indices.append(idx)
            time.sleep(1)  # Wait longer after error
    
    # Save final results
    if save_progress:
        with open(progress_file, 'w') as f:
            json.dump({
                'predictions': predictions,
                'true_labels': true_labels,
                'failed_indices': failed_indices
            }, f)
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Filter out failed predictions for accuracy calculation
    valid_mask = predictions != -1
    valid_predictions = predictions[valid_mask]
    valid_labels = true_labels[valid_mask]
    
    accuracy = (valid_predictions == valid_labels).mean() * 100
    print(f"\n✅ GPT-4o Accuracy: {accuracy:.2f}% ({valid_mask.sum()}/{len(predictions)} valid predictions)")
    
    if failed_indices:
        print(f"⚠️  Failed predictions: {len(failed_indices)}")
    
    return predictions, true_labels


def plot_confusion_matrix(y_true, y_pred, title, filename):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Saved: {filename}")
    plt.close()


def plot_comparison_bar_chart(cnn_acc, gpt4o_acc, cnn_class_acc, gpt4o_class_acc):
    """Plot comparison bar charts"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Overall accuracy comparison
    models = ['CNN\n(Trained)', 'GPT-4o\n(Zero-Shot)']
    accuracies = [cnn_acc, gpt4o_acc]
    colors = ['#2E86AB', '#A23B72']
    
    bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Per-class accuracy comparison
    x = np.arange(len(CLASSES))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, cnn_class_acc, width, label='CNN (Trained)', 
                    color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax2.bar(x + width/2, gpt4o_class_acc, width, label='GPT-4o (Zero-Shot)', 
                    color='#A23B72', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(CLASSES, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.set_ylim([0, 100])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: accuracy_comparison.png")
    plt.close()


def visualize_feature_maps(model, test_dataset, indices, num_samples=3):
    """Visualize CNN feature maps for sample images"""
    print("\n🎨 Visualizing CNN feature maps...")
    
    model.eval()
    normalize = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), 
        (0.2470, 0.2435, 0.2616)
    )
    
    fig, axes = plt.subplots(num_samples, 7, figsize=(18, num_samples*2.5))
    
    for i in range(num_samples):
        idx = indices[i]
        img, label = test_dataset[idx]
        img_normalized = normalize(img).unsqueeze(0).to(device)
        
        # Get feature maps
        with torch.no_grad():
            features = model.get_feature_maps(img_normalized)
        
        # Original image
        img_np = img.numpy().transpose(1, 2, 0)
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Original\n({CLASSES[label]})', fontsize=10)
        axes[i, 0].axis('off')
        
        # Conv1 feature maps (show 3 channels)
        conv1_features = features['conv1'][0].cpu().numpy()
        for j in range(3):
            axes[i, j+1].imshow(conv1_features[j*10], cmap='viridis')
            axes[i, j+1].set_title(f'Conv1 Ch{j*10}', fontsize=9)
            axes[i, j+1].axis('off')
        
        # Conv2 feature maps (show 3 channels)
        conv2_features = features['conv2'][0].cpu().numpy()
        for j in range(3):
            axes[i, j+4].imshow(conv2_features[j*20], cmap='plasma')
            axes[i, j+4].set_title(f'Conv2 Ch{j*20}', fontsize=9)
            axes[i, j+4].axis('off')
    
    plt.suptitle('CNN Feature Maps Visualization', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('cnn_feature_maps.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: cnn_feature_maps.png")
    plt.close()


def visualize_failure_cases(test_dataset, indices, cnn_preds, gpt4o_preds, true_labels, num_samples=10):
    """Visualize failure cases for both models"""
    print("\n🎨 Visualizing failure cases...")
    
    # Find cases where models disagree or both fail
    cnn_wrong = cnn_preds != true_labels
    gpt4o_wrong = gpt4o_preds != true_labels
    
    # Interesting cases:
    # 1. Both wrong
    both_wrong = np.where(cnn_wrong & gpt4o_wrong)[0]
    # 2. CNN correct, GPT-4o wrong
    cnn_right_gpt_wrong = np.where(~cnn_wrong & gpt4o_wrong)[0]
    # 3. GPT-4o correct, CNN wrong
    gpt_right_cnn_wrong = np.where(cnn_wrong & ~gpt4o_wrong)[0]
    
    fig, axes = plt.subplots(3, min(5, len(both_wrong)), figsize=(15, 9))
    if len(both_wrong) == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot both wrong
    for i in range(min(5, len(both_wrong))):
        if i >= len(both_wrong):
            axes[0, i].axis('off')
            continue
        
        idx_in_subset = both_wrong[i]
        idx = indices[idx_in_subset]
        img, _ = test_dataset[idx]
        img_np = img.numpy().transpose(1, 2, 0)
        
        axes[0, i].imshow(img_np)
        axes[0, i].set_title(
            f'True: {CLASSES[true_labels[idx_in_subset]]}\n'
            f'CNN: {CLASSES[cnn_preds[idx_in_subset]]}\n'
            f'GPT: {CLASSES[gpt4o_preds[idx_in_subset]]}',
            fontsize=8, color='red'
        )
        axes[0, i].axis('off')
    
    axes[0, 0].set_ylabel('Both Wrong', fontsize=10, fontweight='bold')
    
    # Plot CNN right, GPT-4o wrong
    for i in range(min(5, len(cnn_right_gpt_wrong))):
        if i >= len(cnn_right_gpt_wrong):
            axes[1, i].axis('off')
            continue
        
        idx_in_subset = cnn_right_gpt_wrong[i]
        idx = indices[idx_in_subset]
        img, _ = test_dataset[idx]
        img_np = img.numpy().transpose(1, 2, 0)
        
        axes[1, i].imshow(img_np)
        axes[1, i].set_title(
            f'True: {CLASSES[true_labels[idx_in_subset]]}\n'
            f'CNN: ✓ {CLASSES[cnn_preds[idx_in_subset]]}\n'
            f'GPT: ✗ {CLASSES[gpt4o_preds[idx_in_subset]]}',
            fontsize=8
        )
        axes[1, i].axis('off')
    
    axes[1, 0].set_ylabel('CNN ✓ GPT ✗', fontsize=10, fontweight='bold')
    
    # Plot GPT-4o right, CNN wrong
    for i in range(min(5, len(gpt_right_cnn_wrong))):
        if i >= len(gpt_right_cnn_wrong):
            axes[2, i].axis('off')
            continue
        
        idx_in_subset = gpt_right_cnn_wrong[i]
        idx = indices[idx_in_subset]
        img, _ = test_dataset[idx]
        img_np = img.numpy().transpose(1, 2, 0)
        
        axes[2, i].imshow(img_np)
        axes[2, i].set_title(
            f'True: {CLASSES[true_labels[idx_in_subset]]}\n'
            f'CNN: ✗ {CLASSES[cnn_preds[idx_in_subset]]}\n'
            f'GPT: ✓ {CLASSES[gpt4o_preds[idx_in_subset]]}',
            fontsize=8
        )
        axes[2, i].axis('off')
    
    axes[2, 0].set_ylabel('CNN ✗ GPT ✓', fontsize=10, fontweight='bold')
    
    plt.suptitle('Interesting Failure Cases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('failure_cases.png', dpi=150, bbox_inches='tight')
    print("📊 Saved: failure_cases.png")
    plt.close()


def generate_report(cnn_preds, gpt4o_preds, true_labels):
    """Generate comprehensive comparison report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON REPORT")
    print("="*80)
    
    # Overall metrics
    cnn_acc = (cnn_preds == true_labels).mean() * 100
    
    # Filter valid GPT-4o predictions
    valid_mask = gpt4o_preds != -1
    gpt4o_acc = (gpt4o_preds[valid_mask] == true_labels[valid_mask]).mean() * 100
    
    print(f"\n📊 OVERALL ACCURACY")
    print(f"   CNN (Trained):        {cnn_acc:.2f}%")
    print(f"   GPT-4o (Zero-Shot):   {gpt4o_acc:.2f}%")
    print(f"   Difference:           {cnn_acc - gpt4o_acc:+.2f}%")
    
    # Per-class metrics
    print(f"\n📊 PER-CLASS METRICS")
    print("-"*80)
    print(f"{'Class':<12} {'CNN Acc':<10} {'GPT Acc':<10} {'CNN P/R/F1':<25} {'GPT P/R/F1':<25}")
    print("-"*80)
    
    cnn_class_acc = []
    gpt4o_class_acc = []
    
    for i, class_name in enumerate(CLASSES):
        mask = true_labels == i
        
        # CNN metrics for this class
        cnn_class_correct = (cnn_preds[mask] == true_labels[mask]).sum()
        cnn_class_total = mask.sum()
        cnn_class_accuracy = (cnn_class_correct / cnn_class_total * 100) if cnn_class_total > 0 else 0
        cnn_class_acc.append(cnn_class_accuracy)
        
        # GPT-4o metrics for this class
        valid_class_mask = mask & valid_mask
        gpt_class_correct = (gpt4o_preds[valid_class_mask] == true_labels[valid_class_mask]).sum()
        gpt_class_total = valid_class_mask.sum()
        gpt_class_accuracy = (gpt_class_correct / gpt_class_total * 100) if gpt_class_total > 0 else 0
        gpt4o_class_acc.append(gpt_class_accuracy)
        
        print(f"{class_name:<12} {cnn_class_accuracy:>6.2f}%    {gpt_class_accuracy:>6.2f}%")
    
    # Classification reports
    print(f"\n📊 CNN CLASSIFICATION REPORT")
    print("-"*80)
    print(classification_report(true_labels, cnn_preds, target_names=CLASSES, digits=4))
    
    print(f"\n📊 GPT-4o CLASSIFICATION REPORT")
    print("-"*80)
    print(classification_report(true_labels[valid_mask], gpt4o_preds[valid_mask], 
                                target_names=CLASSES, digits=4))
    
    # Save report to file
    with open('comparison_report.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("CIFAR-10: CNN vs GPT-4o Vision Comparison Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"Overall Accuracy:\n")
        f.write(f"  CNN (Trained):        {cnn_acc:.2f}%\n")
        f.write(f"  GPT-4o (Zero-Shot):   {gpt4o_acc:.2f}%\n")
        f.write(f"  Difference:           {cnn_acc - gpt4o_acc:+.2f}%\n\n")
        f.write("CNN Classification Report:\n")
        f.write(classification_report(true_labels, cnn_preds, target_names=CLASSES, digits=4))
        f.write("\n\nGPT-4o Classification Report:\n")
        f.write(classification_report(true_labels[valid_mask], gpt4o_preds[valid_mask], 
                                      target_names=CLASSES, digits=4))
    
    print("💾 Report saved to comparison_report.txt")
    
    return cnn_acc, gpt4o_acc, cnn_class_acc, gpt4o_class_acc


def main():
    """Main comparison pipeline"""
    print("="*80)
    print("CIFAR-10: Trained CNN vs. GPT-4o Vision Zero-Shot Comparison")
    print("="*80)
    
    # Load test dataset (no normalization for raw images)
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Create stratified test subset
    if os.path.exists('stratified_test_indices.json'):
        print("\n📂 Loading existing stratified test indices...")
        with open('stratified_test_indices.json', 'r') as f:
            indices = json.load(f)
        print(f"✅ Loaded {len(indices)} indices")
    else:
        indices = create_stratified_test_subset(test_dataset, num_samples=2000)
    
    # Load trained CNN
    if not os.path.exists('best_cnn_model.pth'):
        print("\n❌ Error: best_cnn_model.pth not found!")
        print("Please run train_cnn.py first to train the model.")
        return
    
    cnn_model = load_trained_cnn('best_cnn_model.pth')
    
    # Evaluate CNN
    cnn_preds, true_labels = evaluate_cnn(cnn_model, test_dataset, indices)
    
    # Evaluate GPT-4o
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: OPENAI_API_KEY not found in environment")
        print("GPT-4o evaluation will be skipped.")
        print("To include GPT-4o evaluation, create a .env file with:")
        print("OPENAI_API_KEY=your_key_here")
        gpt4o_preds = None
    else:
        gpt4o_preds, _ = evaluate_gpt4o(test_dataset, indices, api_key)
    
    # Generate visualizations and report
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS AND REPORT")
    print("="*80)
    
    # Confusion matrices
    plot_confusion_matrix(true_labels, cnn_preds, 
                         'CNN Confusion Matrix', 
                         'cnn_confusion_matrix.png')
    
    if gpt4o_preds is not None:
        valid_mask = gpt4o_preds != -1
        plot_confusion_matrix(true_labels[valid_mask], gpt4o_preds[valid_mask], 
                             'GPT-4o Confusion Matrix', 
                             'gpt4o_confusion_matrix.png')
    
    # Generate comprehensive report
    if gpt4o_preds is not None:
        cnn_acc, gpt4o_acc, cnn_class_acc, gpt4o_class_acc = generate_report(
            cnn_preds, gpt4o_preds, true_labels
        )
        
        # Comparison charts
        plot_comparison_bar_chart(cnn_acc, gpt4o_acc, cnn_class_acc, gpt4o_class_acc)
        
        # Failure cases
        visualize_failure_cases(test_dataset, indices, cnn_preds, gpt4o_preds, true_labels)
    
    # Feature maps
    visualize_feature_maps(cnn_model, test_dataset, indices, num_samples=3)
    
    print("\n" + "="*80)
    print("✅ COMPARISON COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  📊 cnn_confusion_matrix.png")
    if gpt4o_preds is not None:
        print("  📊 gpt4o_confusion_matrix.png")
        print("  📊 accuracy_comparison.png")
        print("  📊 failure_cases.png")
    print("  📊 cnn_feature_maps.png")
    print("  📄 comparison_report.txt")
    print("\n🎉 All done!")


if __name__ == "__main__":
    main()

