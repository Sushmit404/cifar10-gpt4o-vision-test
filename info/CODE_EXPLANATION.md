# Code Explanation: CIFAR-10 Classification Project

This document provides a detailed high-level explanation of how the code works, including the mathematical foundations of softmax and cross-entropy loss.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [train_cnn_orig.py - Baseline CNN](#train_cnn_origpy---baseline-cnn)
4. [train_cnn_32.py & train_cnn_224.py - Improved CNN](#train_cnn_32py--train_cnn_224py---improved-cnn)
5. [cifar10_gpt4o_test.py & evaluate_gpt4o.py - GPT-4o Evaluation](#cifar10_gpt4o_testpy--evaluate_gpt4opy---gpt-4o-evaluation)
6. [Slide Bullet Points](#slide-bullet-points)

---

## Overview

This project compares three approaches to CIFAR-10 image classification:

1. **Baseline CNN** (`train_cnn_orig.py`): Simple 2-layer convolutional network
2. **Improved CNN** (`train_cnn_32.py`, `train_cnn_224.py`): ResNet-style architecture with advanced techniques
3. **GPT-4o Vision** (`cifar10_gpt4o_test.py`, `evaluate_gpt4o.py`): Zero-shot classification using OpenAI's multimodal model

All models are evaluated on the same **stratified subset of 2,000 test images** (200 per class) for fair comparison.

---

## Mathematical Foundations

### Softmax Function

**Purpose**: Converts raw model outputs (logits) into a probability distribution over classes.

**Mathematical Definition**:
```
σ(z)_i = exp(z_i) / Σ_j exp(z_j)
```

Where:
- `z` = vector of raw logits (one per class)
- `z_i` = logit for class i
- `σ(z)_i` = probability that the input belongs to class i

**Properties**:
- All probabilities sum to 1: `Σ_i σ(z)_i = 1`
- All probabilities are non-negative: `σ(z)_i ≥ 0`
- Higher logits → higher probabilities

**Numerical Stability**:
To prevent `exp()` overflow with large logits, we subtract the maximum:
```
z_stable = z - max(z)
σ(z)_i = exp(z_stable_i) / Σ_j exp(z_stable_j)
```

This doesn't change the result because:
```
exp(z_i - c) / Σ exp(z_j - c) = exp(z_i)/exp(c) / (Σ exp(z_j)/exp(c))
                              = exp(z_i) / Σ exp(z_j)
```

**Example**:
```
Input logits: [2.0, 1.0, 0.1]
After subtracting max (2.0): [0.0, -1.0, -1.9]
exp([0.0, -1.0, -1.9]) = [1.0, 0.368, 0.150]
Sum = 1.518
Probabilities: [1.0/1.518, 0.368/1.518, 0.150/1.518] = [0.659, 0.242, 0.099]
```

### Cross-Entropy Loss

**Purpose**: Measures how far the predicted probability distribution is from the true distribution.

**Mathematical Definition**:
```
L = -Σ_i y_i * log(p_i)
```

Where:
- `y_i` = true probability of class i (1 for correct class, 0 for others)
- `p_i` = predicted probability of class i (from softmax)

**For Single Correct Class** (our case):
If the true class is `c`, then `y_c = 1` and `y_i = 0` for all `i ≠ c`:
```
L = -log(p_c)
```

**Intuition**:
- If `p_c = 1.0` (perfect prediction): `L = -log(1.0) = 0` ✅
- If `p_c = 0.5` (50% confident): `L = -log(0.5) = 0.693`
- If `p_c = 0.1` (10% confident): `L = -log(0.1) = 2.303`
- If `p_c → 0` (very wrong): `L → ∞` ❌

**Label Smoothing** (used in improved CNN):
Instead of hard labels `y = [0, 0, 1, 0, ...]`, we use:
```
y_smooth = [ε/(K-1), ε/(K-1), 1-ε, ε/(K-1), ...]
```
Where `ε = 0.1` and `K = 10` classes.

This prevents overconfidence and improves generalization.

**Batch Loss**:
For a batch of `N` samples:
```
L_batch = (1/N) * Σ_n L_n
```

---

## train_cnn_orig.py - Baseline CNN

### Architecture

**Simple 2-Layer Convolutional Network**:

```
Input: (batch, 3, 32, 32)  # RGB images, 32×32 pixels

Conv Block 1:
  Conv2d(3 → 32 channels, kernel=3, padding=1)
  BatchNorm2d(32)
  ReLU
  MaxPool2d(2×2)  # 32×32 → 16×16

Conv Block 2:
  Conv2d(32 → 64 channels, kernel=3, padding=1)
  BatchNorm2d(64)
  ReLU
  MaxPool2d(2×2)  # 16×16 → 8×8

Fully Connected Head:
  Flatten: 64 × 8 × 8 = 4,096 features
  Linear(4,096 → 128)
  ReLU
  Dropout(0.5)
  Linear(128 → 10)  # 10 classes

Output: (batch, 10)  # Raw logits
```

**Total Parameters**: ~1.2M

### Training Pipeline

1. **Data Loading**:
   - Loads CIFAR-10 training set (50,000 images)
   - Applies data augmentation: RandomHorizontalFlip, RandomCrop
   - Normalizes with CIFAR-10 mean/std: `(0.4914, 0.4822, 0.4465)` / `(0.2470, 0.2435, 0.2616)`
   - Batches data (default batch_size=128)

2. **Training Loop** (for each epoch):
   ```
   For each batch:
     a. Forward pass: images → model → logits
     b. Compute loss: CrossEntropyLoss(logits, labels)
     c. Backward pass: compute gradients
     d. Update weights: optimizer.step()
     e. Track accuracy: count correct predictions
   ```

3. **Evaluation**:
   - Loads stratified test subset (2,000 images)
   - Runs model in eval mode (disables dropout)
   - Computes accuracy, confusion matrix, precision/recall/F1

4. **Optimization**:
   - **Optimizer**: Adam (adaptive learning rate)
   - **Learning Rate**: 0.001
   - **Scheduler**: StepLR (decays LR every 30 epochs)
   - **Loss**: CrossEntropyLoss (no label smoothing)

5. **Early Stopping**:
   - Monitors validation accuracy
   - Stops if no improvement for `patience` epochs
   - Saves best model checkpoint

### Key Features

- Simple architecture (easy to understand)
- Standard data augmentation
- Basic regularization (dropout, batch norm)
- **Expected Accuracy**: ~65-70%

---

## train_cnn_32.py & train_cnn_224.py - Improved CNN

### Architecture

**ResNet-Style Deep Network with Residual Connections**:

```
Input: (batch, 3, 32×32 or 224×224)

Initial Conv:
  Conv2d(3 → 64, kernel=3, padding=1)
  BatchNorm2d(64)
  ReLU

Layer 1 (2 Residual Blocks):
  ResidualBlock(64 → 64) × 2
  MaxPool2d(2×2)  # 32×32 → 16×16 (or 224×224 → 112×112)

Layer 2 (2 Residual Blocks):
  ResidualBlock(64 → 128, stride=2)  # Downsample
  ResidualBlock(128 → 128)
  MaxPool2d(2×2)  # 16×16 → 8×8 (or 112×112 → 56×56)

Layer 3 (2 Residual Blocks):
  ResidualBlock(128 → 256, stride=2)  # Downsample
  ResidualBlock(256 → 256)

Fully Connected:
  Flatten: 256 × 2×2 = 1,024 (32×32) or 256 × 14×14 = 50,176 (224×224)
  Linear(1,024/50,176 → 256)
  ReLU
  Dropout(0.5)
  Linear(256 → 10)

Output: (batch, 10)  # Raw logits
```

### Residual Block

**Key Innovation**: Skip connections that allow gradients to flow directly.

```
ResidualBlock(x):
  out = Conv2d(x) → BatchNorm → ReLU
  out = Conv2d(out) → BatchNorm
  out = Dropout2d(out)
  out = out + shortcut(x)  # Skip connection
  return ReLU(out)
```

**Why It Works**:
- Enables training of very deep networks (solves vanishing gradient problem)
- Allows model to learn identity mappings when needed
- Improves gradient flow during backpropagation

**Parameters**:
- **32×32 model**: ~1.2M parameters
- **224×224 model**: ~15.7M parameters (mostly in FC layer)

### Training Pipeline

1. **Data Loading**:
   - **32×32**: Native CIFAR-10 size
   - **224×224**: Upscales images using bilinear interpolation (matches GPT-4o preprocessing)
   - **Advanced Augmentation**:
     - RandomHorizontalFlip
     - RandomCrop (with padding)
     - ColorJitter (brightness, contrast, saturation, hue)
     - RandomRotation (±10°)
     - RandomAffine (translation, scaling)
     - RandomErasing (randomly removes patches)

2. **Training Loop**:
   ```
   For each epoch:
     For each batch:
       a. Forward: images → model → logits
       b. Loss: CrossEntropyLoss(logits, labels, label_smoothing=0.1)
       c. Backward: compute gradients
       d. Gradient Clipping: clip_grad_norm_(max_norm=1.0)  # Prevents exploding gradients
       e. Optimizer: Adam(weight_decay=1e-4)  # L2 regularization
       f. Scheduler: CosineAnnealingLR  # Smooth LR decay
   ```

3. **Optimization**:
   - **Optimizer**: Adam with weight decay (L2 regularization)
   - **Learning Rate**: 
     - 32×32: 0.001
     - 224×224: 0.0001 (10× lower, needed for stability)
   - **Scheduler**: CosineAnnealingLR (smooth cosine decay)
   - **Loss**: CrossEntropyLoss with label smoothing (ε=0.1)
   - **Gradient Clipping**: Prevents exploding gradients

4. **Early Stopping** (Dual Mechanism):
   - **Improvement-based**: Stops if no improvement for `patience` epochs
   - **Flatline Detection**: Stops if accuracy unchanged for 20 epochs (prevents wasted training)

5. **Weight Initialization**:
   - **Conv/Linear**: Kaiming normal (He initialization) - good for ReLU
   - **BatchNorm**: Constant(1) for weight, Constant(0) for bias

### Key Improvements Over Baseline

1. **Deeper Architecture**: ResNet-style blocks enable learning complex features
2. **Residual Connections**: Solve vanishing gradient problem
3. **Advanced Augmentation**: More robust to variations
4. **Label Smoothing**: Prevents overconfidence
5. **Gradient Clipping**: Stabilizes training
6. **Weight Decay**: L2 regularization reduces overfitting
7. **Better Initialization**: Kaiming normal for faster convergence

**Expected Accuracy**:
- **32×32**: ~75-80%
- **224×224**: ~76-81% (slight improvement, but much slower)

---

## cifar10_gpt4o_test.py & evaluate_gpt4o.py - GPT-4o Evaluation

### Overview

These scripts evaluate GPT-4o Vision (OpenAI's multimodal model) on CIFAR-10 using **zero-shot classification** (no training on CIFAR-10).

### cifar10_gpt4o_test.py (Simple Version)

**Purpose**: Quick test on a few sample images.

**Pipeline**:

1. **Load CIFAR-10 Test Data**:
   ```python
   test_data = datasets.CIFAR10(root="./data", train=False, download=True)
   ```

2. **Select Sample Images**:
   - Default: 5 images (indices [0, 1, 2, 3, 4])
   - Can specify custom indices

3. **Image Preprocessing**:
   ```python
   # Convert PyTorch tensor → PIL Image
   arr = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
   img = Image.fromarray(arr)
   
   # Upscale to 224×224 (GPT-4o works better on larger images)
   img = img.resize((224, 224), Image.BILINEAR)
   
   # Encode as PNG bytes
   buffer = io.BytesIO()
   img.save(buffer, format='PNG')
   png_bytes = buffer.getvalue()
   ```

4. **API Call**:
   ```python
   client = OpenAI(api_key=api_key)
   response = client.chat.completions.create(
       model="gpt-4o",
       messages=[{
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
       }],
       max_tokens=10
   )
   ```

5. **Parse Response**:
   ```python
   prediction = response.choices[0].message.content.strip().lower()
   ```

6. **Calculate Accuracy**:
   ```python
   correct = sum(1 for t, p in zip(true_labels, predictions) if t == p)
   accuracy = (correct / len(true_labels)) * 100
   ```

### evaluate_gpt4o.py (Full Evaluation)

**Purpose**: Evaluate on full 2,000-image stratified subset.

**Key Features**:

1. **Batch Processing**:
   - Processes all 2,000 images
   - Progress bar with tqdm
   - Rate limiting (0.5s delay between requests)

2. **Checkpoint System**:
   - Saves results every 50 images
   - Can resume if interrupted
   - Saves to JSON: `gpt4o_results.json`

3. **Error Handling**:
   - Retry logic (3 attempts per image)
   - Handles API errors gracefully
   - Logs failures for debugging

4. **Stratified Subset**:
   - Uses same `stratified_subset_2000.json` as CNN
   - Ensures fair comparison (200 images per class)

5. **Results Format**:
   ```json
   {
       "model": "gpt-4o",
       "total_images": 2000,
       "correct": 1936,
       "accuracy": 0.968,
       "predictions": [
           {"index": 0, "true_label": "airplane", "predicted": "airplane", "correct": true},
           ...
       ],
       "per_class_accuracy": {...},
       "confusion_matrix": [...]
   }
   ```

### How GPT-4o Works (High-Level)

1. **Vision Encoder**: Converts image to embeddings (similar to CNN feature extraction)
2. **Language Model**: Processes text prompt + image embeddings
3. **Text Generation**: Generates class label as text
4. **Zero-Shot**: No fine-tuning on CIFAR-10 (uses pre-trained knowledge)

**Why It's So Good**:
- Trained on massive dataset (images + text)
- Understands semantic relationships
- Can handle low-resolution images (with upscaling)
- **Expected Accuracy**: ~96-97%

---

## Slide Bullet Points

### train_cnn_orig

**Architecture**:
- Simple 2-layer CNN: Conv → Pool → Conv → Pool → FC
- 32×32 input, 10-class output
- ~1.2M parameters

**Training**:
- Adam optimizer, LR=0.001
- Standard data augmentation (flip, crop)
- Cross-entropy loss (no label smoothing)
- Early stopping based on validation accuracy

**Results**:
- Baseline accuracy: ~65-70%
- Fast training (~15 seconds/epoch)
- Good starting point for comparison

**Key Features**:
- Minimal architecture (easy to understand)
- Standard regularization (dropout, batch norm)
- No advanced techniques

---

### train_cnn (32×32 & 224×224)

**Architecture**:
- ResNet-style deep network with residual blocks
- Skip connections solve vanishing gradient problem
- 32×32: ~1.2M params | 224×224: ~15.7M params

**Training Improvements**:
- Label smoothing (ε=0.1) prevents overconfidence
- Gradient clipping (max_norm=1.0) stabilizes training
- Weight decay (L2 regularization) reduces overfitting
- Cosine annealing LR scheduler (smooth decay)
- Advanced augmentation (rotation, color jitter, erasing)

**Optimization**:
- 32×32: LR=0.001, batch_size=128
- 224×224: LR=0.0001 (10× lower for stability), batch_size=64
- Kaiming normal initialization (faster convergence)

**Early Stopping**:
- Dual mechanism: improvement-based + flatline detection
- Stops if no improvement for 20 epochs OR accuracy unchanged for 20 epochs

**Results**:
- 32×32: ~75-80% accuracy (significant improvement)
- 224×224: ~76-81% accuracy (slight gain, much slower)
- Training time: 32×32 ~26 min, 224×224 ~2-3 hours

**Key Innovations**:
- Residual connections enable deeper networks
- Comprehensive data augmentation improves generalization
- Label smoothing + gradient clipping = stable training

---

### cifar10_gpt4o_test

**Approach**:
- Zero-shot classification (no training on CIFAR-10)
- Uses OpenAI GPT-4o Vision API
- Pre-trained multimodal model (images + text)

**Preprocessing**:
- Upscales 32×32 → 224×224 using bilinear interpolation
- Converts to PNG format, base64-encoded
- Same preprocessing as CNN 224×224 for fair comparison

**API Interaction**:
- Sends image + text prompt to GPT-4o
- Prompt: "Classify as exactly one of: airplane, automobile, ..."
- Returns text label (parsed to match CIFAR-10 classes)

**Evaluation**:
- Same stratified subset (2,000 images, 200 per class)
- Batch processing with progress tracking
- Checkpoint system (resume if interrupted)
- Error handling with retry logic

**Results**:
- Accuracy: ~96-97% (best performance)
- Fast inference (~0.5s per image via API)
- No training required (uses pre-trained knowledge)

**Why It Works So Well**:
- Massive pre-training data (internet-scale)
- Understands semantic relationships
- Can handle low-res images with upscaling
- Multimodal understanding (images + text context)

**Limitations**:
- Requires API access (cost per request)
- Slower than local inference
- No fine-tuning on CIFAR-10 (zero-shot only)

---

## Summary Comparison

| Model | Parameters | Accuracy | Training Time | Notes |
|-------|-----------|----------|---------------|-------|
| Baseline CNN | 1.2M | ~65-70% | ~26 min | Simple, fast |
| Improved 32×32 | 1.2M | ~75-80% | ~26 min | ResNet-style, better |
| Improved 224×224 | 15.7M | ~76-81% | ~2-3 hours | Larger, slightly better |
| GPT-4o Vision | ~1.8T | ~96-97% | 0 (zero-shot) | Best, but API-dependent |

**Key Takeaway**: GPT-4o's massive pre-training gives it superior performance, but the improved CNNs show significant gains over the baseline through architectural improvements and better training techniques.

