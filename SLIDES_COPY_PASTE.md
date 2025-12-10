# SLIDE-BY-SLIDE COPY/PASTE VERSION
## Just copy each slide's content into your presentation tool

---

## SLIDE 1 (TITLE)

**Title:** Can a Custom CNN Compete with GPT-4o Vision?

**Subtitle:** Trained vs. Zero-Shot Image Classification on CIFAR-10

**Course:** CS260 Final Project

**Date:** December 2025

---

## SLIDE 2 (MOTIVATION)

**Title:** The Rise of Giant AI Models

**Bullet Points:**
• GPT-4o: ~1.8 TRILLION parameters
• Trained on billions of internet images
• Zero-shot: No training needed for new tasks

**Question Box:**
"Can a small, custom-trained CNN compete with massive pre-trained models?"

**Visual:** Show GPT-4o logo vs small CNN diagram

---

## SLIDE 3 (DATASET)

**Title:** The Dataset: CIFAR-10

**Left Column - Stats:**
• 60,000 total images
• 50,000 training / 10,000 test
• Image size: 32×32 pixels (TINY!)
• 10 classes

**Right Column - Classes:**
✈️ airplane | 🚗 automobile | 🐦 bird | 🐱 cat | 🦌 deer
🐕 dog | 🐸 frog | 🐴 horse | 🚢 ship | 🚛 truck

**Visual:** Grid of CIFAR-10 sample images

---

## SLIDE 4 (METHODOLOGY)

**Title:** Fair Comparison: Stratified Sampling

**The Problem:**
• CIFAR-10 test set = 10,000 images
• Testing all with API = expensive & slow
• Need smaller BUT representative subset

**Our Solution - Stratified Random Sampling:**

1. Group 10,000 images by class label (0-9)
2. Randomly select 200 indices per class
3. Use seed=42 for reproducibility
4. Save indices to JSON file

**Key Numbers:**
| Original | Subset | Per Class | Seed |
|----------|--------|-----------|------|
| 10,000 | 2,000 | 200 | 42 |

**Technical Detail (mention briefly):**
```python
np.random.choice(class_indices, size=200, replace=False)
# Saved to: stratified_subset_2000.json
```

**Why This Matters:**
✅ Both models tested on IDENTICAL 2,000 images
✅ Perfect class balance (no bias)
✅ Reproducible (seed=42 → same indices)
✅ Affordable (~$3 API cost, not $15)

**Visual:** Bar chart showing 200 images per class

---

## SLIDE 5 (BASELINE CNN) — Friend Presents

**Title:** Method 1: Our Baseline CNN

**Architecture Diagram:**
```
32×32 Image
    ↓
Conv Layer 1 (32 filters)
    ↓
Conv Layer 2 (64 filters)
    ↓
Fully Connected
    ↓
10 Class Output
```

**Specs:**
• 2 convolutional layers
• ~1.2 million parameters
• Trained on 50,000 images
• Training time: 26 minutes

---

## SLIDE 6 (BASELINE RESULTS) — Friend Presents

**Title:** Baseline CNN Results

**BIG NUMBER:** 71.55% Accuracy

**Best/Worst Table:**
| Best Classes | Worst Classes |
|--------------|---------------|
| automobile: 86.5% | cat: 42.5% |
| truck: 83.0% | bird: 54.5% |
| horse: 81.5% | deer: 60.0% |

**Pain Point:** "56 cats were predicted as dogs! 😿"

**Visual:** Confusion matrix heatmap

---

## SLIDE 7 (GPT-4o METHOD) — You Present

**Title:** Method 2: GPT-4o Vision (Zero-Shot)

**Key Point:** NO TRAINING REQUIRED!

**How It Works:**
1. Take 32×32 image
2. Convert to PNG
3. Send to OpenAI API
4. Get class label back

**The Prompt:**
"Classify this image as exactly one of: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. Return only the label."

**Specs:**
• 1.8 trillion parameters
• Cost: ~$0.0008 per image (~$3 total!)
• Zero training on CIFAR-10

---

## SLIDE 8 (GPT-4o RESULTS) — You Present

**Title:** GPT-4o Results

**BIG NUMBER:** 96.75% Accuracy (1,935/2,000)

**Per-Class Accuracy:**
| Class | Accuracy | Class | Accuracy |
|-------|----------|-------|----------|
| airplane | 97.5% | dog | 93.5% |
| automobile | 97.5% | frog | 93.5% |
| bird | 95.0% | horse | 98.0% |
| cat | 94.0% | ship | 99.0% |
| deer | 96.5% | truck | 98.5% |

**Visual:** Confusion matrix (mostly diagonal!)

---

## SLIDE 9 (FUN FACTS) — You Present

**Title:** 💸 Fun Fact: The API Adventure

**The Numbers:**
| Stat | Value |
|------|-------|
| API calls | 4,000 |
| Total tokens | 1,196,000 |
| Total cost | **$3.04** ☕ |
| Cost per image | $0.00076 |

**Why So Cheap?**
• Tiny 32×32 images = few tokens
• Short responses ("cat", "dog")
• GPT-4o: $2.50 per 1M tokens

**Interesting Finding:**
We tested BOTH:
• 224×224 upscaled → 96.75%
• 32×32 raw → 96.75%

"Upscaling had NO effect!"

**Quote:** "4,000 API calls for $3... less than a coffee!"

---

## SLIDE 10 (THE GAP)

**Title:** 😱 The Problem: A Huge Gap!

**Comparison:**
| Model | Accuracy |
|-------|----------|
| Our CNN | 71.55% |
| GPT-4o | 96.75% |
| **GAP** | **25%** |

**Key Stat:**
GPT-4o has 1,500,000× more parameters!

**But we asked:**
"What if we IMPROVE our CNN?"

---

## SLIDE 11 (IMPROVEMENTS + ARCHITECTURE)

**Title:** 🔨 How We Improved the CNN

**Improvements Applied:**
| Category | Techniques |
|----------|------------|
| Architecture | ResNet blocks, skip connections, 6 conv layers, channels 64→128→256 |
| Augmentation | ColorJitter, RandomRotation(±10°), RandomErasing, RandomCrop |
| Training | Label smoothing, CosineAnnealingLR, Early stopping |

**Architecture (simplified):**
```
32×32 → Conv(64) → ResNet×2(64) → ResNet×2(128) → ResNet×2(256) → FC → 10 classes
```

**Key Insight:** Skip connections let gradients flow → enables deeper training

**Training:** 43 min, 119 epochs, GPU was 🔥 HOT!

---

## SLIDE 12 (RESULTS + COMPARISON)

**Title:** 🎉 Results: 71% → 92%

**Per-Class Improvement:**
| Class | Before | After | Δ |
|-------|--------|-------|---|
| cat | 42.5% | 84.0% | +41.5% 🚀 |
| bird | 54.5% | 87.5% | +33.0% |
| deer | 60.0% | 93.0% | +33.0% |

**Final Showdown:**
| Model | Accuracy | Parameters | Cost |
|-------|----------|------------|------|
| Baseline CNN | 71.55% | 1.2M | Free |
| **Improved CNN** | **92.10%** | 2.3M | Free |
| GPT-4o | 96.75% | 1.8T | ~$3 |

**Gap: 25% → 5% (closed 80%!)** 🎯

---

## SLIDE 13 (KEY FINDINGS)

**Title:** 💡 Key Findings

| Finding | Insight |
|---------|---------|
| Zero-shot ≠ Unbeatable | GPT-4o wins by only ~5% |
| Architecture Matters | ResNet + skip connections: +5-8% |
| Augmentation is Crucial | ColorJitter, RandomErasing: +3-5% |
| Upscaling = No Effect | GPT-4o: 32×32 = 224×224 (both 96.75%) |

**Common Confusions (All Models):**
• Cat ↔ Dog (fur at 32×32)
• Bird ↔ Airplane (flying objects)
• Auto ↔ Truck (vehicle shapes)

---

## SLIDE 14 (CONCLUSIONS + FUTURE)

**Title:** 🚀 Conclusions & Future Work

**Main Takeaways:**
✅ Custom CNNs CAN compete with massive models
✅ Systematic improvements work (+21%)
✅ Trade-offs: CNN = free & fast | GPT-4o = accurate & zero-shot
✅ Zero-shot is powerful but not unbeatable

**Future Work (If 6 More Months):**
• Models: Vision Transformers, attention, transfer learning
• Experiments: Other LLMs (Claude, Gemini), CIFAR-100
• Analysis: Grad-CAM visualizations

---

## SLIDE 15 (REFERENCES + THANK YOU)

**Title:** 📚 References & Thank You

**References:**
1. CIFAR-10 - Krizhevsky (2009) - cs.toronto.edu/~kriz/cifar.html
2. GPT-4o - OpenAI (2024) - platform.openai.com/docs/models/gpt-4o
3. ResNet - He et al. (2016) - arxiv.org/abs/1512.03385
4. Our Code - github.com/Sushmit404/cifar10-gpt4o-vision-test

**Key Numbers:**
| Model | Accuracy |
|-------|----------|
| Baseline CNN | 71.55% |
| **Improved CNN** | **92.10%** |
| **GPT-4o** | **96.75%** |
| **Gap Closed** | **80%** 🎯 |

**Questions?**

---

# TIMING GUIDE (8 minutes)

| Slides | Who | Minutes |
|--------|-----|---------|
| 1-4 (Intro, Data) | Either | 1.5 |
| 5-6 (Baseline CNN) | Friend | 1.5 |
| 7-9 (GPT-4o + Fun) | You | 1.5 |
| 10-11 (Improvements) | Either | 1.5 |
| 12-15 (Results, Conclusions) | Either | 2.0 |

**TOTAL: 8 minutes (15 slides)**

---

# VISUALS NEEDED

Use these from your project:

1. `subset_distribution.png` - Class distribution
2. `results_cnn/cnn_confusion_matrix.png` - Baseline confusion
3. `results_cnn_32_20251209_132041/cnn_confusion_matrix.png` - Improved confusion
4. `gpt_4o_32_visualizations/` - GPT-4o visualizations
5. `results_cnn_32_20251209_132041/cnn_summary_dashboard.png` - Summary

---

# CASUAL TALKING POINTS

Drop these naturally during presentation:

💬 "Fun fact: those 4,000 API calls cost about $3... less than a coffee!"

💬 "The GPU was NOT happy during those 43 minutes of training"

💬 "At 32×32 pixels, honestly even WE couldn't tell some cats from dogs"

💬 "We thought upscaling would help GPT-4o, but nope - same accuracy!"

💬 "1.8 trillion parameters vs 1.2 million... that's like comparing a library to a post-it note"


