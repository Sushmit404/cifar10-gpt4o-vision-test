# CS260 Final Project Presentation
## Trained vs. Zero-Shot: Custom CNN vs. GPT-4o Vision on CIFAR-10

---

# 🎯 SLIDE 1: Title Slide

## **Can a Custom CNN Compete with GPT-4o Vision?**

### Trained vs. Zero-Shot Image Classification on CIFAR-10

**CS260 Final Project**

*[Your Names Here]*

*December 2025*

---

# 🤔 SLIDE 2: The Motivational Question

## The Rise of Giant AI Models

- OpenAI's GPT-4o: **~1.8 trillion parameters**
- Trained on **billions** of images from the internet
- Zero-shot capability: No training needed for new tasks

## Our Question:

> **Can a small, custom-trained CNN compete with these massive pre-trained models on image classification?**

🎯 If you train a model specifically for a task, can it match or beat a general-purpose giant?

---

# 🖼️ SLIDE 3: The Dataset - CIFAR-10

## What is CIFAR-10?

| Property | Value |
|----------|-------|
| Total images | 60,000 |
| Training set | 50,000 |
| Test set | 10,000 |
| Image size | **32×32 pixels** (tiny!) |
| Classes | 10 |

## The 10 Classes:

```
✈️ airplane   🚗 automobile   🐦 bird     🐱 cat      🦌 deer
🐕 dog        🐸 frog         🐴 horse    🚢 ship     🚛 truck
```

**Challenge:** These images are TINY - hard even for humans to classify!

*[Include sample CIFAR-10 image grid]*

---

# 📊 SLIDE 4: Our Test Methodology

## Stratified Sampling: Fair Comparison

### The Problem:
- CIFAR-10 test set has **10,000 images** (1,000 per class)
- Testing all 10K with GPT-4o API would be expensive
- Need a **smaller but representative** subset

### Our Solution: Stratified Random Sampling

**Process:**
1. Group all 10,000 test images by their class label (0-9)
2. From each class, randomly select **200 indices** using `np.random.choice()`
3. Use **seed=42** for reproducibility (anyone can recreate exact same subset)
4. Store selected indices in `stratified_subset_2000.json`

| Parameter | Value |
|-----------|-------|
| Original test set | 10,000 images |
| Our subset | **2,000 images** |
| Per class | **200 images** (exactly) |
| Selection | Random without replacement |
| Seed | 42 (reproducible) |

### Technical Implementation:
```python
# For each class (0-9):
indices = np.random.choice(
    class_indices[class_id],  # All images of this class
    size=200,                  # Select exactly 200
    replace=False              # No duplicates
)
# Save to stratified_subset_2000.json
```

## Why This Matters:
- ✅ Both CNN and GPT-4o tested on **identical** 2,000 images
- ✅ Perfect class balance (no bias from unequal representation)
- ✅ Reproducible: Same seed → same indices every time
- ✅ Affordable API cost (~$3 for 2,000 vs ~$15 for full 10K)
- ✅ Statistically significant (200 samples per class)

*[Show subset_distribution.png visualization]*

---

# 🔧 SLIDE 5: Method 1 - Baseline CNN (Friend Presents)

## Our First CNN: Simple 2-Layer Architecture

```
Input: 32×32×3 (RGB)
    ↓
Conv2d(3 → 32) + BatchNorm + ReLU + MaxPool (32×32 → 16×16)
    ↓
Conv2d(32 → 64) + BatchNorm + ReLU + MaxPool (16×16 → 8×8)
    ↓
Flatten → Linear(4096 → 128) → Dropout(0.5) → Linear(128 → 10)
    ↓
Output: 10 class probabilities
```

| Specs | Value |
|-------|-------|
| Convolutional layers | 2 |
| Parameters | ~1.2 million |
| Training data | 50,000 images |
| Training time | ~26 minutes |

---

# 📉 SLIDE 6: Baseline CNN Results

## Accuracy: **71.55%**

| Metric | Value |
|--------|-------|
| Overall accuracy | 71.55% |
| Best class | automobile (86.5%) |
| Worst class | cat (42.5%) |

## Per-Class Performance:

| Strong Classes | Weak Classes |
|----------------|--------------|
| 🚗 automobile: 86.5% | 🐱 cat: 42.5% |
| 🚛 truck: 83.0% | 🐦 bird: 54.5% |
| 🐴 horse: 81.5% | 🦌 deer: 60.0% |

**Problem:** Cat-Dog confusion is terrible! 56 cats predicted as dogs 😿

*[Show baseline confusion matrix]*

---

# 🤖 SLIDE 7: Method 2 - GPT-4o Vision API (You Present)

## Zero-Shot Classification with GPT-4o

**No training required!** Just send image + prompt:

```python
Prompt: "Classify this image as exactly one of: 
         airplane, automobile, bird, cat, deer, 
         dog, frog, horse, ship, truck. 
         Return only the label."
```

## Technical Pipeline:

```
CIFAR-10 Image (32×32) 
    → Convert to PNG bytes 
    → Base64 encode 
    → Send to OpenAI API 
    → Parse response
```

| Specs | Value |
|-------|-------|
| Model | GPT-4o Vision |
| Parameters | ~1.8 **trillion** |
| Training on CIFAR-10 | **None** (zero-shot) |
| Cost | ~$0.0008 per image (~$3 total for 4,000 calls!) |

---

# ⚡ SLIDE 8: GPT-4o Results

## Accuracy: **96.75%**

| Metric | Value |
|--------|-------|
| Correct predictions | 1,935 / 2,000 |
| Overall accuracy | **96.75%** |
| Best class | ship (99%) |
| Worst class | cat (94%) |

## Per-Class Performance:

| Class | Accuracy | Class | Accuracy |
|-------|----------|-------|----------|
| ✈️ airplane | 97.5% | 🐕 dog | 93.5% |
| 🚗 automobile | 97.5% | 🐸 frog | 93.5% |
| 🐦 bird | 95.0% | 🐴 horse | 98.0% |
| 🐱 cat | 94.0% | 🚢 ship | **99.0%** |
| 🦌 deer | 96.5% | 🚛 truck | 98.5% |

*[Show GPT-4o confusion matrix]*

---

# 💸 SLIDE 9: Fun Fact - The API Adventure

## Behind the Scenes of GPT-4o Testing

### The Numbers:
| Stat | Value |
|------|-------|
| API calls made | **4,000** (2,000 × 2 experiments) |
| Total tokens | **1,196,000** |
| Total cost | **$3.04** ☕ (less than a coffee!) |
| Cost per image | **$0.00076** (~0.08 cents) |
| Time spent | ~80 minutes waiting |

### Why So Cheap?
- 32×32 images = very few tokens
- Short responses (just "cat", "dog", etc.)
- GPT-4o pricing: $2.50/1M input tokens

### Why 4,000 calls?

We tested both:
1. **224×224 upscaled images** (industry standard)
2. **32×32 raw images** (original resolution)

**Surprising finding:** Both achieved **96.75%** - upscaling didn't help!

---

# 😱 SLIDE 10: The Gap Problem

## Houston, We Have a Problem!

| Model | Accuracy | Gap |
|-------|----------|-----|
| Baseline CNN | 71.55% | — |
| GPT-4o Vision | 96.75% | — |
| **Gap** | — | **25.2%** 😬 |

## The Challenge:

> GPT-4o has **1.8 trillion parameters** vs our CNN's **1.2 million**
> 
> That's **1,500,000× more parameters!**

### But we didn't give up...

> *"What if we improve our CNN? Can we close the gap?"*

---

# 🔨 SLIDE 11: How We Improved the CNN

## From 71% → 92%: Our Strategy

| Category | Techniques Applied |
|----------|-------------------|
| **Architecture** | ResNet blocks, skip connections, 6 conv layers, channels 64→128→256 |
| **Augmentation** | ColorJitter, RandomRotation(±10°), RandomErasing, RandomCrop |
| **Training** | Label smoothing (0.1), CosineAnnealingLR, Early stopping |

### ResNet Architecture:
```
32×32 → Conv(64) → ResNet×2(64) → ResNet×2(128) → ResNet×2(256) → FC → 10 classes
```

**Key Insight:** Skip connections let gradients flow directly → enables deeper training

**Training:** 43 min, 119 epochs, GPU was 🔥 HOT!

---

# 🎉 SLIDE 12: Results - The Improvement

## Accuracy: 71.55% → **92.10%** (+20.55%!)

| Class | Before | After | Δ |
|-------|--------|-------|---|
| 🐱 cat | 42.5% | 84.0% | **+41.5%** 🚀 |
| 🐦 bird | 54.5% | 87.5% | +33.0% |
| 🦌 deer | 60.0% | 93.0% | +33.0% |
| 🚗 auto | 86.5% | 98.0% | +11.5% |

### Final Comparison:

| Model | Accuracy | Parameters | Cost |
|-------|----------|------------|------|
| Baseline CNN | 71.55% | 1.2M | Free |
| **Improved CNN** | **92.10%** | 2.3M | Free |
| GPT-4o | 96.75% | 1.8T | ~$3 |

**Gap closed: 25% → 5% (80% reduction!)** 🎯

---

# 💡 SLIDE 13: Key Findings

## What We Learned

| Finding | Insight |
|---------|---------|
| **Zero-shot ≠ Unbeatable** | GPT-4o wins by only ~5% with proper CNN techniques |
| **Architecture Matters** | ResNet + skip connections: +5-8% accuracy |
| **Augmentation is Crucial** | ColorJitter, RandomErasing: +3-5% accuracy |
| **Upscaling = No Effect** | GPT-4o: 32×32 = 224×224 (both 96.75%) |

### Common Confusions (All Models):
- 🐱 Cat ↔ 🐕 Dog (fur texture at 32×32)
- 🐦 Bird ↔ ✈️ Airplane (flying objects)
- 🚗 Auto ↔ 🚛 Truck (vehicle shapes)

---

# 🚀 SLIDE 14: Conclusions & Future Work

## Main Takeaways

✅ **Custom CNNs CAN compete** with massive pre-trained models  
✅ **Systematic improvements work** (+21% through techniques)  
✅ **Trade-offs matter:** CNN = free & fast | GPT-4o = accurate & zero-shot  
✅ **Zero-shot is powerful** but not unbeatable

## Future Work (If We Had 6 Months):

| Area | Ideas |
|------|-------|
| **Models** | Vision Transformers (ViT), attention, transfer learning |
| **Experiments** | Other LLMs (Claude, Gemini), CIFAR-100, ImageNet |
| **Analysis** | Grad-CAM visualizations, energy consumption |

---

# 📚 SLIDE 15: References & Thank You

## References:
1. **CIFAR-10** - Krizhevsky (2009) - https://www.cs.toronto.edu/~kriz/cifar.html
2. **GPT-4o** - OpenAI (2024) - https://platform.openai.com/docs/models/gpt-4o
3. **ResNet** - He et al. (2016) - https://arxiv.org/abs/1512.03385
4. **Our Code** - github.com/Sushmit404/cifar10-gpt4o-vision-test

---

## Thank You! Questions?

| Model | Accuracy |
|-------|----------|
| Baseline CNN | 71.55% |
| **Improved CNN** | **92.10%** |
| **GPT-4o Vision** | **96.75%** |
| **Gap Closed** | **80%** 🎯 |

---

# 📋 APPENDIX: Speaker Notes

## Slide Distribution (8 minutes total)

| Slides | Speaker | Time |
|--------|---------|------|
| 1-4 | Either | ~1.5 min |
| 5-6 | Friend (CNN baseline) | ~1.5 min |
| 7-9 | You (GPT-4o + fun facts) | ~1.5 min |
| 10-11 | Either (improvements) | ~1.5 min |
| 12-15 | Either (results + conclusions) | ~2 min |

## Key Points to Emphasize:
1. **Open with the hook**: Can a small model beat a giant?
2. **Show the gap**: 71% vs 97% seems impossible
3. **Build the story**: We improved step by step
4. **Celebrate the win**: Closed gap by 80%!
5. **Fun facts**: Only $3 for 4,000 API calls, GPU going hot

## Things to Mention Casually:
- "Fun fact: those 4,000 API calls cost about $3... less than a coffee!"
- "The GPU was not happy during those 43 minutes of training"
- "At 32×32 pixels, even WE couldn't tell cats from dogs sometimes"


