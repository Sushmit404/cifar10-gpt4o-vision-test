"""
Analyze CNN errors and identify which classes are most confused.
"""

import torch
import numpy as np
from train_cnn import CustomCNN, final_evaluation, create_stratified_test_subset, get_data_loaders, CLASSES
from torchvision import datasets, transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading model and data (upscaled to 224×224 to match GPT-4o)...")
import sys
sys.stdout.flush()

# Use 224×224 input size to match GPT-4o Vision
INPUT_SIZE = 224
train_loader, test_dataset = get_data_loaders(input_size=INPUT_SIZE)
test_indices = create_stratified_test_subset(test_dataset)

# Load trained model
print("Loading model...")
sys.stdout.flush()
model = CustomCNN(input_size=INPUT_SIZE).to(device)
checkpoint = torch.load('results_cnn/best_cnn_model.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Running evaluation (this may take a moment)...")
sys.stdout.flush()

# Run evaluation without printing (we'll do our own analysis)
model.eval()
all_predictions = []
all_labels = []

normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

from tqdm import tqdm
with torch.no_grad():
    for idx in tqdm(test_indices, desc='Evaluating', leave=False):
        img, label = test_dataset[idx]
        img_normalized = normalize(img).unsqueeze(0).to(device)
        outputs = model(img_normalized)
        _, predicted = outputs.max(1)
        all_predictions.append(predicted.item())
        all_labels.append(label)

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Compute confusion matrix
cm = np.zeros((10, 10), dtype=np.int32)
for true_label, pred_label in zip(all_labels, all_predictions):
    cm[true_label, pred_label] += 1

print('\n' + '='*70)
print('CNN ERROR ANALYSIS')
print('='*70)

# Overall accuracy
overall_acc = np.trace(cm) / np.sum(cm) * 100
print(f'\nOverall Accuracy: {overall_acc:.2f}%')
print(f'Total Errors: {np.sum(cm) - np.trace(cm)}/{np.sum(cm)}')

# Per-class accuracy (recall)
print('\n' + '-'*70)
print('PER-CLASS ACCURACY (Recall)')
print('-'*70)
print(f"{'Class':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<12} {'Errors':<10}")
print('-'*70)

class_accuracies = []
for i, cls in enumerate(CLASSES):
    correct = cm[i, i]
    total = np.sum(cm[i, :])
    errors = total - correct
    acc = correct / total * 100 if total > 0 else 0
    class_accuracies.append((cls, acc, errors))
    print(f"{cls:<15} {correct:<10} {total:<10} {acc:>10.2f}%   {errors:<10}")

# Sort by accuracy (worst first)
class_accuracies.sort(key=lambda x: x[1])

print('\n' + '-'*70)
print('CLASSES RANKED BY ACCURACY (worst to best)')
print('-'*70)
for cls, acc, errors in class_accuracies:
    print(f"{cls:<15}: {acc:>6.2f}% ({errors} errors)")

# Most confused pairs
print('\n' + '-'*70)
print('MOST CONFUSED CLASS PAIRS')
print('-'*70)
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

confused_pairs = []
for _ in range(10):
    idx = np.unravel_index(np.argmax(cm_no_diag), cm.shape)
    if cm_no_diag[idx] > 0:
        confused_pairs.append((CLASSES[idx[0]], CLASSES[idx[1]], cm_no_diag[idx]))
        cm_no_diag[idx] = 0
    else:
        break

print(f"{'True Label':<15} -> {'Predicted':<15} {'Count':<10}")
print('-'*70)
for true_cls, pred_cls, count in confused_pairs:
    print(f"{true_cls:<15} -> {pred_cls:<15} {count:<10}")

# Error distribution
print('\n' + '-'*70)
print('ERROR DISTRIBUTION')
print('-'*70)
print(f"{'Class':<15} {'Errors Made':<15} {'Errors Received':<15}")
print('-'*70)
for i, cls in enumerate(CLASSES):
    errors_made = np.sum(cm[i, :]) - cm[i, i]  # How many this class got wrong
    errors_received = np.sum(cm[:, i]) - cm[i, i]  # How many were misclassified as this
    print(f"{cls:<15} {errors_made:<15} {errors_received:<15}")

print('\n' + '='*70)

