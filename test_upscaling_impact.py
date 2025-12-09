"""
Test the impact of upscaling CIFAR-10 images on CNN performance.

This script evaluates whether upscaling 32×32 images to higher resolutions
(e.g., 64×64, 128×128) improves classification accuracy, especially for
fine-grained distinctions like cat vs dog.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from train_cnn import CustomCNN, CLASSES, device

def create_upscaled_cnn(input_size=64, num_classes=10):
    """
    Create a CNN adapted for upscaled input size.
    
    Args:
        input_size: Input image size (32, 64, 128, etc.)
        num_classes: Number of classes
    """
    # Calculate feature map size after conv layers
    # 2 pooling layers with stride 2: size / 4
    feature_size = input_size // 4
    
    class UpscaledCNN(nn.Module):
        def __init__(self):
            super(UpscaledCNN, self).__init__()
            
            # Conv block 1
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.pool1 = nn.MaxPool2d(2, 2)
            
            # Conv block 2
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool2 = nn.MaxPool2d(2, 2)
            
            # FC head - adapt to feature size
            fc_input_size = 64 * feature_size * feature_size
            self.fc1 = nn.Linear(fc_input_size, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)
            
        def forward(self, x):
            x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
            x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 64 * (x.size(2) * x.size(3)))
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
    
    return UpscaledCNN()

def evaluate_upscaling_impact():
    """
    Compare model performance at different input resolutions.
    """
    print("="*70)
    print("UPSCALING IMPACT ANALYSIS")
    print("="*70)
    
    # Test different resolutions
    resolutions = [32, 64, 128]
    results = {}
    
    # Load test data
    print("\nLoading CIFAR-10 test set...")
    test_transform_base = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, 
                                   transform=test_transform_base)
    
    # Use stratified subset for fair comparison
    import json
    with open('stratified_subset_2000.json', 'r') as f:
        data = json.load(f)
        if isinstance(data, dict) and 'indices' in data:
            test_indices = data['indices']
        else:
            test_indices = data
    
    test_indices = test_indices[:200]  # Use subset for faster testing
    
    print(f"\nTesting on {len(test_indices)} images...")
    print(f"Resolutions to test: {resolutions}")
    
    for res in resolutions:
        print(f"\n{'='*70}")
        print(f"Testing at {res}×{res} resolution")
        print('='*70)
        
        # Create upscaling transform
        if res == 32:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        else:
            # Upscale using different methods
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.resize((res, res), Image.BILINEAR)),  # Upscale
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        
        # Create model for this resolution
        model = create_upscaled_cnn(input_size=res).to(device)
        
        # Quick training (just a few epochs for comparison)
        print(f"Training model for {res}×{res}...")
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4) if res == 32 else transforms.Lambda(lambda x: x.resize((res, res), Image.BILINEAR)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, 
                                      transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, 
                                                   shuffle=True, num_workers=2)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Train for 5 epochs (quick test)
        model.train()
        for epoch in range(5):
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"  Epoch {epoch+1}/5, Loss: {running_loss/len(train_loader):.4f}")
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(10)}
        class_total = {i: 0 for i in range(10)}
        
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        
        with torch.no_grad():
            for idx in test_indices:
                img, label = test_dataset[idx]
                
                # Upscale if needed
                if res > 32:
                    img_pil = transforms.ToPILImage()(img)
                    img_pil = img_pil.resize((res, res), Image.BILINEAR)
                    img = transforms.ToTensor()(img_pil)
                
                img_normalized = normalize(img).unsqueeze(0).to(device)
                outputs = model(img_normalized)
                _, predicted = outputs.max(1)
                
                total += 1
                if predicted.item() == label:
                    correct += 1
                    class_correct[label] += 1
                class_total[label] += 1
        
        accuracy = 100. * correct / total
        results[res] = {
            'accuracy': accuracy,
            'class_acc': {CLASSES[i]: 100.*class_correct[i]/class_total[i] 
                          if class_total[i] > 0 else 0 
                          for i in range(10)}
        }
        
        print(f"\n{res}×{res} Results:")
        print(f"  Overall Accuracy: {accuracy:.2f}%")
        print(f"  Cat Accuracy: {results[res]['class_acc']['cat']:.2f}%")
        print(f"  Dog Accuracy: {results[res]['class_acc']['dog']:.2f}%")
        print(f"  Bird Accuracy: {results[res]['class_acc']['bird']:.2f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Upscaling Impact")
    print("="*70)
    print(f"{'Resolution':<15} {'Overall Acc':<15} {'Cat Acc':<15} {'Dog Acc':<15} {'Bird Acc':<15}")
    print("-"*70)
    for res in resolutions:
        r = results[res]
        print(f"{res}×{res:<11} {r['accuracy']:>13.2f}% {r['class_acc']['cat']:>13.2f}% "
              f"{r['class_acc']['dog']:>13.2f}% {r['class_acc']['bird']:>13.2f}%")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    baseline = results[32]['accuracy']
    for res in [64, 128]:
        if res in results:
            improvement = results[res]['accuracy'] - baseline
            print(f"\n{res}×{res} vs 32×32:")
            print(f"  Accuracy change: {improvement:+.2f}%")
            print(f"  Cat improvement: {results[res]['class_acc']['cat'] - results[32]['class_acc']['cat']:+.2f}%")
            print(f"  Dog improvement: {results[res]['class_acc']['dog'] - results[32]['class_acc']['dog']:+.2f}%")
    
    return results

if __name__ == "__main__":
    results = evaluate_upscaling_impact()

