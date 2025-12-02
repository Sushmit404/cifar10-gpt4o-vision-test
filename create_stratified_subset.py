"""
Create Stratified Test Subset for CIFAR-10 Evaluation
CS260 Final Project - GPT-4o Vision Track

This script creates a balanced 2,000-image subset (200 per class)
for fair comparison between CNN and GPT-4o Vision models.
"""

import numpy as np
import json
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt


# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']


def download_cifar10_test():
    """Download CIFAR-10 test dataset"""
    print("📦 Downloading CIFAR-10 test dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✅ Test dataset size: {len(test_data)}")
    return test_data


def create_stratified_subset(test_data, images_per_class=200, seed=42):
    """
    Create a stratified subset with equal images per class
    
    Args:
        test_data: CIFAR-10 test dataset
        images_per_class: Number of images to sample per class (default 200)
        seed: Random seed for reproducibility (default 42)
    
    Returns:
        dict with 'indices', 'class_distribution', and 'total_images'
    """
    print(f"\n🎲 Creating stratified subset with seed={seed}")
    print(f"Target: {images_per_class} images per class")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Group images by class
    class_indices = defaultdict(list)
    
    print("📊 Organizing images by class...")
    for idx in range(len(test_data)):
        _, label = test_data[idx]
        class_indices[label].append(idx)
    
    # Display class distribution in original dataset
    print("\n📈 Original test set distribution:")
    for class_id, indices in sorted(class_indices.items()):
        print(f"  Class {class_id} ({CLASSES[class_id]:>10}): {len(indices):>4} images")
    
    # Sample equally from each class
    stratified_indices = []
    class_distribution = {}
    
    print(f"\n🔀 Sampling {images_per_class} images from each class...")
    for class_id in range(10):
        available_indices = class_indices[class_id]
        
        # Check if we have enough images
        if len(available_indices) < images_per_class:
            print(f"⚠️  Warning: Class {class_id} has only {len(available_indices)} images!")
            sampled = available_indices
        else:
            # Randomly sample without replacement
            sampled = np.random.choice(
                available_indices, 
                size=images_per_class, 
                replace=False
            ).tolist()
        
        stratified_indices.extend(sampled)
        class_distribution[CLASSES[class_id]] = len(sampled)
    
    # Sort indices for consistent ordering
    stratified_indices.sort()
    
    # Create result dictionary
    result = {
        'indices': stratified_indices,
        'class_distribution': class_distribution,
        'total_images': len(stratified_indices),
        'images_per_class': images_per_class,
        'seed': seed,
        'classes': CLASSES
    }
    
    print(f"\n✅ Stratified subset created!")
    print(f"Total images: {len(stratified_indices)}")
    
    return result


def verify_subset(test_data, subset_info):
    """
    Verify that the subset is properly balanced
    
    Args:
        test_data: CIFAR-10 test dataset
        subset_info: Dictionary returned by create_stratified_subset
    """
    print("\n🔍 Verifying subset balance...")
    
    indices = subset_info['indices']
    class_counts = defaultdict(int)
    
    # Count actual distribution
    for idx in indices:
        _, label = test_data[idx]
        class_counts[label] += 1
    
    # Display verification results
    print("\n📊 Actual subset distribution:")
    all_balanced = True
    for class_id in range(10):
        count = class_counts[class_id]
        expected = subset_info['images_per_class']
        status = "✅" if count == expected else "❌"
        print(f"  {status} Class {class_id} ({CLASSES[class_id]:>10}): {count:>3} images (expected {expected})")
        if count != expected:
            all_balanced = False
    
    if all_balanced:
        print("\n✅ Perfect balance achieved!")
    else:
        print("\n⚠️  Warning: Imbalanced distribution detected!")
    
    return all_balanced


def save_subset_info(subset_info, filename='stratified_subset_2000.json'):
    """
    Save subset indices to JSON file for reproducibility
    
    Args:
        subset_info: Dictionary with subset information
        filename: Output filename (default: stratified_subset_2000.json)
    """
    print(f"\n💾 Saving subset info to {filename}...")
    
    with open(filename, 'w') as f:
        json.dump(subset_info, f, indent=2)
    
    print(f"✅ Saved {len(subset_info['indices'])} indices")
    print(f"📄 File: {filename}")


def visualize_distribution(subset_info, filename='subset_distribution.png'):
    """
    Create a bar chart showing the class distribution
    
    Args:
        subset_info: Dictionary with subset information
        filename: Output filename for the plot
    """
    print(f"\n📊 Creating distribution visualization...")
    
    classes = subset_info['classes']
    distribution = subset_info['class_distribution']
    
    # Prepare data for plotting
    class_names = [cls for cls in classes]
    counts = [distribution[cls] for cls in classes]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(classes)), counts, color='steelblue', alpha=0.8)
    
    # Customize plot
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    ax.set_title(f'Stratified Subset Distribution ({subset_info["total_images"]} images)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {filename}")


def main():
    """Main function to create and verify stratified subset"""
    print("=" * 60)
    print("  CIFAR-10 Stratified Subset Creation")
    print("  CS260 Final Project - GPT-4o Vision Track")
    print("=" * 60)
    
    # Step 1: Download CIFAR-10 test set
    test_data = download_cifar10_test()
    
    # Step 2: Create stratified subset (200 images per class = 2,000 total)
    subset_info = create_stratified_subset(
        test_data, 
        images_per_class=200,
        seed=42  # Fixed seed for reproducibility
    )
    
    # Step 3: Verify balance
    is_balanced = verify_subset(test_data, subset_info)
    
    # Step 4: Save to JSON file
    save_subset_info(subset_info, 'stratified_subset_2000.json')
    
    # Step 5: Create visualization
    visualize_distribution(subset_info, 'subset_distribution.png')
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total images selected: {subset_info['total_images']}")
    print(f"Images per class: {subset_info['images_per_class']}")
    print(f"Random seed: {subset_info['seed']}")
    print(f"Balanced: {'Yes ✅' if is_balanced else 'No ❌'}")
    print(f"Output files:")
    print(f"  - stratified_subset_2000.json (indices)")
    print(f"  - subset_distribution.png (visualization)")
    print("\n✅ Ready for GPT-4o evaluation!")
    print("=" * 60)


if __name__ == "__main__":
    main()

