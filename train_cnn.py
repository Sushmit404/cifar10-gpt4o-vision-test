"""
CIFAR-10 Custom CNN Training Script
Part 1: Train a custom CNN on CIFAR-10 with GPU support

Evaluation uses a fixed stratified subset of 2,000 test images (200 per class)
to ensure fair comparison with GPT-4o Vision.
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

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
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
    """
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 32x32x3 -> 32x32x32
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 32x32x32 -> 16x16x32
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 16x16x32 -> 16x16x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x16x64 -> 8x8x64
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
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
    indices_file = 'stratified_test_indices.json'
    
    # Load existing indices if available
    if os.path.exists(indices_file):
        print(f"📂 Loading existing stratified test indices from {indices_file}...")
        with open(indices_file, 'r') as f:
            indices = json.load(f)
        print(f"   ✅ Loaded {len(indices)} indices (200 per class)")
        return indices
    
    # Create new stratified subset
    np.random.seed(seed)
    samples_per_class = num_samples // 10
    
    print(f"📊 Creating stratified test subset...")
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
    print(f"   ✅ Stratified subset created and saved to {indices_file}")
    
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
    print("📦 Loading CIFAR-10 dataset...")
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
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_subset_size': len(test_indices),
            }, 'best_cnn_model.pth')
            print(f"✅ Best model saved! (Test Acc: {test_acc:.2f}%)")
    
    print("\n" + "=" * 70)
    print(f"🎉 Training complete! Best test accuracy: {best_acc:.2f}% (on 2,000 stratified test images)")
    
    return history


def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['test_loss'], label='Test Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['test_acc'], label='Test Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("📊 Training history saved to training_history.png")
    plt.close()


def main():
    """Main training pipeline"""
    print("=" * 70)
    print("CIFAR-10 Custom CNN Training")
    print("=" * 70)
    
    # Hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_TEST_SAMPLES = 2000  # Stratified subset for fair comparison with GPT-4o
    
    # Load data
    train_loader, test_dataset = get_data_loaders(BATCH_SIZE)
    
    # Create/load stratified test subset (same 2,000 images used for GPT-4o)
    test_indices = create_stratified_test_subset(test_dataset, num_samples=NUM_TEST_SAMPLES)
    
    # Create model
    print("\n🏗️  Building model...")
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
    print("\n📈 Plotting training history...")
    plot_training_history(history)
    
    # Save training history
    with open('training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("💾 Training history saved to training_history.json")
    
    print("\n✅ Training pipeline complete!")
    print(f"   Best model saved to: best_cnn_model.pth")
    print(f"   Training history plot: training_history.png")
    print(f"   Training history data: training_history.json")
    print(f"   Stratified test indices: stratified_test_indices.json")
    print(f"\n📊 Evaluation used {NUM_TEST_SAMPLES} stratified test images (same as GPT-4o)")


if __name__ == "__main__":
    main()

