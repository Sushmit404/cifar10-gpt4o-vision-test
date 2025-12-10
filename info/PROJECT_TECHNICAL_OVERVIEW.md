# Project Technical Overview: CNN vs GPT-4o Vision on CIFAR-10

## 📋 Brief Overview

**What This Project Does:**
- Compares two image classification approaches on CIFAR-10 (10 classes, 32×32 pixel images)
- **Custom CNN**: Started at ~71% accuracy, improved to ~92% through systematic enhancements
- **GPT-4o Vision**: Zero-shot (no training) → achieves ~97% accuracy
- Both evaluated on the same 2,000 test images (200 per class) for fair comparison

**Key Technical Question:**
Can a locally-trained CNN compete with a massive pre-trained vision-language model? How can we make our CNN stronger to give it a better shot?

**The Journey:**
We started with a simple 2-layer CNN achieving ~71% accuracy. Recognizing GPT-4o's superior performance (~97%), we systematically improved our CNN through architecture enhancements, better training techniques, and advanced data augmentation. Through these improvements, we boosted accuracy from 71% to 92% - a 21 percentage point improvement! While GPT-4o still wins (97%), we significantly closed the gap from 26% to just 5% and learned valuable techniques for improving CNN performance.

**Main Finding:**
GPT-4o performs better (97% vs 92%) but requires API calls and costs money. Our improved CNN is faster, free, and achieves excellent performance - closing the gap from 26% to just 5% difference. We tried our best to make the CNN competitive!

---

## 🔬 Detailed Technical Overview

### 1. Dataset: CIFAR-10

**What is CIFAR-10?**
- 60,000 tiny color images (32×32 pixels)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Standard benchmark for image classification research

**Why 32×32 is Challenging:**
- Very low resolution (most images are 224×224 or larger)
- Hard to see details - models must learn from minimal visual information
- Tests whether models can extract meaningful features from tiny images

**Stratified Test Subset:**
- Created `stratified_subset_2000.json` with exactly 200 images per class
- Ensures fair comparison (both models see same test images)
- Prevents class imbalance from affecting results

---

### 2. Custom CNN Architecture Evolution: From 71% to 92%

**The Challenge:**
Our initial baseline CNN achieved only ~71% accuracy, while GPT-4o was achieving ~97% - a 26 percentage point gap! We recognized that GPT-4o would likely win due to its massive pretraining, but we wanted to see how strong we could make our CNN. We set out to systematically improve our CNN to give it a better shot at competing.

**Baseline Architecture (71.55% accuracy):**
```
Simple 2-Layer CNN:
  Conv2d(3 → 32) + BatchNorm + ReLU + MaxPool  # 32×32 → 16×16
  Conv2d(32 → 64) + BatchNorm + ReLU + MaxPool  # 16×16 → 8×8
  Flatten → Linear(4096 → 128) → Dropout(0.5) → Linear(128 → 10)
  
Total: 2 convolutional layers, ~1.2M parameters
Result: 71.55% accuracy
```

**Improved Architecture (92.10% accuracy):**
**Architecture Type: ResNet-Style with Residual Blocks**

```
Input: 32×32×3 (RGB image)

Initial Conv Layer:
  Conv2d(3 → 64 channels) + BatchNorm + ReLU

Layer 1: Two Residual Blocks (64 → 64)
  Each block: Conv → BatchNorm → Conv → BatchNorm → Skip Connection
  MaxPool: 32×32 → 16×16

Layer 2: Two Residual Blocks (64 → 128)
  First block downsamples (stride=2): 16×16 → 8×8
  MaxPool: 8×8 → 4×4

Layer 3: Two Residual Blocks (128 → 256)
  First block downsamples (stride=2): 4×4 → 2×2

Fully Connected:
  Flatten: 256 × 2 × 2 = 1,024 features
  Linear(1,024 → 256) + ReLU + Dropout(0.5)
  Linear(256 → 10)  # 10 classes

Output: 10 logits (one per class)
```

**Key Technical Components:**

1. **Residual Blocks (Skip Connections)**
   - Problem: Deep networks are hard to train (vanishing gradients)
   - Solution: Skip connections allow gradients to flow directly
   - Formula: `output = F(x) + x` where F(x) is the learned transformation
   - Why it works: Network can learn identity mapping if needed

2. **Batch Normalization**
   - Normalizes activations during training
   - Prevents internal covariate shift
   - Allows higher learning rates
   - Formula: `(x - mean) / sqrt(variance + ε)`

3. **Dropout**
   - Randomly sets 20% of conv features to zero (Dropout2d)
   - Randomly sets 50% of FC neurons to zero (Dropout)
   - Prevents overfitting by forcing network to not rely on specific features

4. **Data Augmentation**
   - RandomHorizontalFlip: Mirrors images horizontally (50% chance)
   - RandomCrop: Crops and pads images (adds variation)
   - ColorJitter: Adjusts brightness/contrast
   - RandomRotation: Rotates ±10 degrees
   - RandomErasing: Randomly erases patches
   - **Why**: Increases effective training data, improves generalization

**Training Process:**

1. **Forward Pass:**
   ```
   Image → Conv Layers → Feature Maps → Flatten → FC Layers → Logits
   ```

2. **Loss Calculation:**
   - CrossEntropyLoss with Label Smoothing (0.1)
   - Label smoothing: Instead of [1, 0, 0, ...], use [0.9, 0.01, 0.01, ...]
   - Prevents overconfidence, improves generalization

3. **Backward Pass (Gradient Descent):**
   - Compute gradients: `∂Loss/∂weights`
   - Update weights: `weights = weights - learning_rate × gradients`
   - Adam optimizer: Adaptive learning rate per parameter

4. **Learning Rate Scheduling:**
   - CosineAnnealingLR: Smoothly decreases learning rate
   - Formula: `lr = lr_min + (lr_max - lr_min) × (1 + cos(π × epoch/max_epochs))/2`

**Training Configuration:**
- Batch size: 128 images per batch
- Learning rate: 0.001 (starts high, decreases over time)
- Epochs: 100-200 (with early stopping)
- Early stopping: Stops if accuracy doesn't improve for 20 epochs
- Weight initialization: Kaiming normal (good for ReLU)

**Improvement Journey: From 71% to 92%**

We systematically applied multiple improvements to boost performance:

**1. Architecture Improvements (+13-21% expected):**
   - **ResNet Blocks with Skip Connections**: Replaced 2 simple conv layers with 3 layers of ResNet blocks (6 conv layers total)
     - Skip connections preserve gradients and enable deeper training
     - Expected: +5-8% accuracy
   - **Increased Model Capacity**: Channels increased from 32→64 to 64→128→256
     - More filters to capture complex patterns
     - Expected: +3-5% accuracy
   - **Enhanced Dropout**: Dropout2d(0.2) after each conv block, Dropout(0.5) in FC layers
     - Better regularization to prevent overfitting
     - Expected: +2-3% accuracy

**2. Data Augmentation (+3-5% expected):**
   - **Advanced Augmentation**: Added ColorJitter, RandomRotation(10°), RandomErasing
   - Increases effective training data diversity
   - Expected: +3-5% accuracy

**3. Training Improvements (+2-4% expected):**
   - **CosineAnnealingLR Scheduler**: Replaced StepLR with smooth cosine decay
     - Better learning rate schedule
     - Expected: +1-2% accuracy
   - **Label Smoothing**: CrossEntropyLoss with label_smoothing=0.1
     - Reduces overconfidence, improves generalization
     - Expected: +1-2% accuracy

**Total Expected Improvement: +18-30%**
**Actual Improvement: 71.55% → 92.10% = +20.55%** ✅

**The Result:**
We tried our best! Through systematic improvements across architecture, data augmentation, and training techniques, we boosted our CNN from 71% to 92% accuracy. While GPT-4o still wins (97%), we closed the gap from 26 percentage points to just 5 percentage points. This demonstrates that with proper techniques, local CNNs can achieve excellent performance and compete closely with massive pre-trained models.

**Results:**
- **Accuracy: 92.10%** on 2,000 test images (up from 71.55%)
- Best classes: automobile (98%), ship (95.5%), airplane (95.5%)
- Worst classes: cat (84%), dog (86%), bird (87.5%)
- Training time: ~43 minutes on GPU
- **Gap to GPT-4o: Reduced from 26% to 5%** (97% vs 92%)

---

### 3. GPT-4o Vision (Zero-Shot)

**What is Zero-Shot?**
- No training on CIFAR-10 data
- Uses knowledge learned from massive internet-scale pretraining
- Just sends image to API and asks: "What class is this?"

**How It Works:**

1. **Image Preparation:**
   - Load 32×32 CIFAR-10 image (no upscaling!)
   - Convert to PNG format
   - Encode as base64 string

2. **API Call:**
   ```python
   Prompt: "Classify this image as exactly one of: airplane, automobile, 
           bird, cat, deer, dog, frog, horse, ship, truck. 
           Return only the label, nothing else."
   
   Image: [base64-encoded 32×32 PNG]
   ```

3. **Model Processing:**
   - GPT-4o Vision is a vision-language model
   - Processes image through vision encoder
   - Uses language understanding to match to class names
   - Returns text label

**Why It Works So Well:**
- **Massive Pretraining**: Trained on billions of images from internet
- **Semantic Understanding**: Understands concepts like "airplane" not just pixels
- **Multimodal**: Can combine visual and textual understanding
- **World Knowledge**: Knows what airplanes, cats, etc. look like from diverse contexts

**Technical Details:**
- Model size: ~1.8 trillion parameters (vs CNN's ~2-3 million)
- Inference: ~0.5-1 second per image (API call)
- Cost: ~$0.01 per image (~$20 for 2,000 images)
- No training needed: Just API calls

**Results:**
- **Accuracy: 96.75%** on 2,000 test images
- Best classes: ship (99%), airplane (97.5%), automobile (97.5%)
- Worst classes: cat (94%), dog (93.5%), frog (93.5%)
- Evaluation time: ~30-60 minutes (API rate limits)

**Key Finding:**
- Works equally well on 32×32 native images vs 224×224 upscaled
- Shows GPT-4o can handle low-resolution images effectively
- No preprocessing needed!

---

### 4. Evaluation Methodology

**Stratified Sampling:**
- Ensures exactly 200 images per class
- Prevents bias from class imbalance
- Same subset used for both models (fair comparison)

**Metrics Computed:**

1. **Accuracy**: `correct_predictions / total_predictions`
2. **Precision**: `true_positives / (true_positives + false_positives)`
   - "Of all predictions as 'cat', how many were actually cats?"
3. **Recall**: `true_positives / (true_positives + false_negatives)`
   - "Of all actual cats, how many did we find?"
4. **F1-Score**: `2 × (precision × recall) / (precision + recall)`
   - Harmonic mean of precision and recall
5. **Confusion Matrix**: Shows which classes are confused with which

**Per-Class Metrics:**
- Each class gets its own precision/recall/F1
- Reveals which classes are hardest (cat, dog, bird)
- Shows where each model struggles

---

### 5. Key Technical Differences

| Aspect | Custom CNN | GPT-4o Vision |
|--------|-----------|---------------|
| **Training** | Trained from scratch on 50K images | Zero-shot (no training) |
| **Architecture** | ResNet-style with residual blocks | Transformer-based vision-language model |
| **Parameters** | ~2-3 million | ~1.8 trillion |
| **Inference Speed** | < 1ms per image | ~0.5-1s per image (API) |
| **Cost** | Free (after training) | ~$0.01 per image |
| **Interpretability** | Can visualize feature maps | Black box |
| **Accuracy** | 92.10% (improved from 71%) | 96.75% |
| **Best At** | Fast, local, free, competitive | High accuracy, zero-shot |
| **Improvement Journey** | 71% → 92% (+21%) | N/A (zero-shot) |

---

### 6. Why CNN Struggles with Some Classes

**Cat vs Dog:**
- Very similar at 32×32 resolution
- Both are small furry animals
- CNN relies on pixel patterns, hard to distinguish
- GPT-4o has semantic understanding from pretraining

**Bird:**
- Small, often in complex backgrounds
- Many bird species look similar at low resolution
- CNN may confuse with airplane (both have wings)

**Why CNN Succeeds:**
- Automobile, ship, truck: Distinct shapes, clear features
- Horse, deer: Different body shapes even at low resolution
- Frog: Distinct green color and shape

---

### 7. Technical Implementation Details

**CNN Training Script (`train_cnn_32.py`):**
- Manual implementations of softmax and cross-entropy (for learning)
- Confusion matrix computation from scratch
- Early stopping with patience
- Checkpoint saving (resume training if interrupted)
- Comprehensive logging and visualization

**GPT-4o Evaluation (`evaluate_gpt4o_no_upscale.py`):**
- Batch processing with progress bars
- Checkpoint system (save every 50 images)
- Error handling and retry logic
- Rate limiting (respects API limits)
- Results saved as JSON with full metrics

**Stratified Subset (`create_stratified_subset.py`):**
- Ensures exactly 200 images per class
- Uses random seed (42) for reproducibility
- Saves indices to JSON file
- Both models use same subset

**Visualization (`generate_cnn_per_class_metrics.py`):**
- Loads saved models without retraining
- Computes per-class metrics from confusion matrix
- Generates visualizations (confusion matrix, per-class performance)
- Works with both old simple CNN and new ResNet CNN

---

### 8. Mathematical Foundations

**Softmax Function:**
```
σ(z)_i = exp(z_i) / Σ_j exp(z_j)
```
- Converts raw logits to probabilities
- All probabilities sum to 1
- Higher logit → higher probability

**Cross-Entropy Loss:**
```
L = -log(p_correct_class)
```
- Penalizes wrong predictions more
- If model is 90% confident and wrong, loss is high
- If model is 50% confident and wrong, loss is medium

**Confusion Matrix:**
```
CM[i,j] = count of images with true label i predicted as j
```
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Used to compute precision, recall, F1

**Precision/Recall:**
```
Precision_i = TP_i / (TP_i + FP_i)  # "How accurate are our predictions?"
Recall_i = TP_i / (TP_i + FN_i)      # "How many did we catch?"
```
- TP = True Positive (correctly predicted as class i)
- FP = False Positive (incorrectly predicted as class i)
- FN = False Negative (actually class i but predicted as something else)

---

### 9. Project Structure

**Key Files:**
- `train_cnn_32.py`: Train CNN on 32×32 images
- `evaluate_gpt4o_no_upscale.py`: Evaluate GPT-4o on 32×32 images (no upscaling)
- `create_stratified_subset.py`: Create balanced test subset
- `generate_cnn_per_class_metrics.py`: Generate detailed metrics from saved models
- `stratified_subset_2000.json`: Fixed test indices (200 per class)

**Results:**
- `results_cnn_32_20251209_132041/`: CNN results (92.10% accuracy)
- `results_gpt4o_32x32/`: GPT-4o results (96.75% accuracy)
- Both have confusion matrices, per-class metrics, visualizations

---

### 10. The Improvement Story: How We Boosted CNN from 71% to 92%

**Starting Point:**
- Baseline CNN: Simple 2-layer architecture
- Initial accuracy: 71.55%
- Gap to GPT-4o: 26 percentage points (97% - 71%)

**Our Strategy:**
Recognizing that GPT-4o would likely win due to massive pretraining, we focused on making our CNN as strong as possible. We systematically applied improvements across architecture, data augmentation, and training techniques.

**Improvements Applied:**

1. **ResNet Blocks with Skip Connections** (+5-8% expected)
   - Replaced simple 2-layer CNN with ResNet-style residual blocks
   - 3 layers of ResNet blocks (6 convolutional layers total)
   - Skip connections preserve gradients and enable deeper training
   - **Why it works**: Allows gradients to flow directly, prevents vanishing gradient problem

2. **Increased Model Capacity** (+3-5% expected)
   - Channels: 64 → 128 → 256 (was 32 → 64)
   - More filters to capture complex patterns
   - **Why it works**: More parameters = more capacity to learn complex features

3. **Enhanced Dropout** (+2-3% expected)
   - Dropout2d(0.2) after each conv block
   - Dropout(0.5) in FC layers
   - Better regularization
   - **Why it works**: Prevents overfitting by forcing network to not rely on specific features

4. **Advanced Data Augmentation** (+3-5% expected)
   - ColorJitter(brightness=0.2, contrast=0.2)
   - RandomRotation(10 degrees)
   - RandomErasing(p=0.1)
   - **Why it works**: Increases effective training data, improves generalization

5. **CosineAnnealingLR Scheduler** (+1-2% expected)
   - Replaced StepLR with CosineAnnealingLR
   - Smoother learning rate decay
   - **Why it works**: Better learning rate schedule helps model converge to better solution

6. **Label Smoothing** (+1-2% expected)
   - CrossEntropyLoss with label_smoothing=0.1
   - Reduces overconfidence
   - **Why it works**: Prevents model from being overconfident, improves generalization

**Results:**
- **Before**: 71.55% accuracy (baseline simple CNN)
- **After**: 92.10% accuracy (improved ResNet CNN)
- **Improvement**: +20.55 percentage points
- **Gap to GPT-4o**: Reduced from 26% to 5% (97% - 92%)

**What We Learned:**
- Systematic improvements can dramatically boost CNN performance
- Architecture matters: ResNet blocks enable deeper, better training
- Data augmentation is crucial for generalization
- Training techniques (scheduling, label smoothing) make a difference
- While GPT-4o still wins, we significantly closed the gap
- Local models can achieve excellent performance with proper techniques

**The Bottom Line:**
We tried our best to make the CNN competitive! Through systematic improvements, we boosted accuracy by 21 percentage points. While GPT-4o's massive pretraining gives it an edge (97% vs 92%), our improved CNN demonstrates that local models can achieve excellent performance with the right techniques.

---

### 11. Key Takeaways

**Technical Insights:**
1. **Residual connections** enable deeper networks and better training
2. **Data augmentation** significantly improves generalization
3. **Label smoothing** prevents overconfidence
4. **Stratified sampling** ensures fair evaluation
5. **Zero-shot models** can outperform trained models with enough pretraining
6. **Systematic improvements** can dramatically boost CNN performance (+20% in our case)

**Practical Insights:**
1. CNN is fast and free but requires training
2. GPT-4o is accurate but slow and expensive
3. For production: Use CNN if speed/cost matters
4. For research: GPT-4o shows upper bound of performance
5. Both approaches have their place depending on requirements
6. **With proper techniques, local CNNs can compete closely with massive APIs**

**What Makes This Project Interesting:**
- Direct comparison of trained vs zero-shot
- Same evaluation set ensures fair comparison
- Shows trade-offs between different approaches
- Demonstrates that local models can compete with massive APIs
- **Shows the journey of improving a model from 71% to 92% through systematic enhancements**

---

## 🎯 Summary

This project demonstrates that:
1. **Custom CNNs** can achieve excellent performance (92%) with proper architecture and training - **improved from 71% through systematic enhancements**
2. **GPT-4o Vision** achieves even better performance (97%) without any training
3. The **trade-off** is speed/cost vs accuracy
4. Both models struggle with similar classes (cat/dog) at low resolution
5. **Stratified evaluation** ensures fair comparison
6. **Systematic improvements** can dramatically boost CNN performance (+21 percentage points in our case)

**The Improvement Story:**
We started with a simple 2-layer CNN achieving 71% accuracy. Recognizing GPT-4o's superior performance (97%), we systematically improved our CNN through:
- ResNet-style architecture with residual blocks
- Increased model capacity (64→128→256 channels)
- Advanced data augmentation
- Better training techniques (label smoothing, cosine annealing)
- Enhanced regularization

**Result**: Boosted accuracy from 71% to 92% - closing the gap from 26% to just 5%! We tried our best to make the CNN competitive, and while GPT-4o still wins, we significantly closed the gap and learned valuable techniques for improving CNN performance.

The technical complexity lies in:
- Understanding residual blocks and why they work
- Knowing when to use data augmentation
- Understanding precision vs recall
- Knowing how to evaluate models fairly
- Understanding the difference between training and zero-shot approaches
- **Systematically applying improvements to boost performance**

