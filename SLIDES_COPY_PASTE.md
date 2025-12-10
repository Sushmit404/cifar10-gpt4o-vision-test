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

**Key Numbers:**
• 2,000 test images (from 10,000)
• Exactly 200 per class
• Same images for BOTH models

**Why?**
✅ Fair comparison
✅ No class bias
✅ Affordable (~$20 API cost)

**Visual:** Pie chart showing equal class distribution

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
• Cost: ~$0.01 per image
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

**Numbers:**
• 4,000 API calls total
• ~$40 spent
• 80 minutes of waiting

**Interesting Finding:**
We tested BOTH:
• 224×224 upscaled → 96.75%
• 32×32 raw → 96.75%

"Upscaling had NO effect!"

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

## SLIDE 11 (IMPROVEMENTS)

**Title:** 🔨 Making Our CNN Stronger

**Architecture Improvements:**
• ResNet-style residual blocks
• Skip connections
• More layers (2 → 6)
• More channels (64 → 128 → 256)

**Data Augmentation:**
• Random flips & crops
• Color jitter
• Random rotation (±10°)
• Random erasing

**Training Techniques:**
• Label smoothing (0.1)
• Cosine learning rate schedule
• Early stopping (patience=20)

---

## SLIDE 12 (IMPROVED ARCHITECTURE)

**Title:** The Improved CNN Architecture

**Diagram:**
```
32×32 Input
    ↓
Initial Conv (64 filters)
    ↓
ResNet Block × 2 (64 channels)
    ↓
ResNet Block × 2 (128 channels)
    ↓
ResNet Block × 2 (256 channels)
    ↓
Fully Connected → 10 classes
```

**What's a Residual Block?**
"Skip connections let gradients flow directly!"

**Visual:** Show skip connection diagram

---

## SLIDE 13 (TRAINING FUN FACT)

**Title:** 🔥 Training the Beast

**Stats:**
• 43 minutes of training
• 119 epochs (early stopped)
• GPU temperature: HOT! 🌡️

**The Journey:**
• Epoch 1: 45%
• Epoch 50: 85%
• Epoch 99: 92.10% (best!)

---

## SLIDE 14 (IMPROVED RESULTS)

**Title:** 🎉 Improved CNN Results

**BIG NUMBER:** 92.10% Accuracy (+20.55%!)

**Before/After Table:**
| Class | Before | After | Improvement |
|-------|--------|-------|-------------|
| cat | 42.5% | 84.0% | +41.5% 🚀 |
| bird | 54.5% | 87.5% | +33.0% |
| deer | 60.0% | 93.0% | +33.0% |
| automobile | 86.5% | 98.0% | +11.5% |

---

## SLIDE 15 (FINAL COMPARISON)

**Title:** 📊 The Final Showdown

**Comparison Table:**
| Metric | Baseline CNN | Improved CNN | GPT-4o |
|--------|--------------|--------------|--------|
| Accuracy | 71.55% | 92.10% | 96.75% |
| Parameters | 1.2M | 2.3M | 1.8T |
| Training | 26 min | 43 min | None |
| Cost/image | Free | Free | $0.01 |

**Gap Reduction:**
• Before: 25%
• After: 5%
• **Closed 80% of the gap!**

---

## SLIDE 16 (BAR CHART)

**Title:** Accuracy by Class

**Create bar chart with these values:**

| Class | Baseline | Improved | GPT-4o |
|-------|----------|----------|--------|
| airplane | 78.5% | 95.5% | 97.5% |
| automobile | 86.5% | 98.0% | 97.5% |
| bird | 54.5% | 87.5% | 95.0% |
| cat | 42.5% | 84.0% | 94.0% |
| deer | 60.0% | 93.0% | 96.5% |
| dog | 69.0% | 86.0% | 93.5% |
| frog | 80.5% | 93.5% | 93.5% |
| horse | 81.5% | 93.0% | 98.0% |
| ship | 79.5% | 95.5% | 99.0% |
| truck | 83.0% | 95.0% | 98.5% |

**Visual:** Grouped bar chart (3 bars per class)

---

## SLIDE 17 (CONFUSIONS)

**Title:** 🔍 Where Models Struggle

**The Cat-Dog Problem:**
| Model | Cat → Dog | Dog → Cat |
|-------|-----------|-----------|
| Baseline | 56 | 33 |
| Improved | 12 | 16 |
| GPT-4o | 5 | 6 |

**Why?** "At 32×32 pixels, even WE can't tell them apart!"

**Other Confusions:**
• Bird ↔ Airplane (both fly!)
• Automobile ↔ Truck
• Deer ↔ Horse

**Visual:** Show confused image examples

---

## SLIDE 18 (KEY FINDINGS)

**Title:** 💡 Key Findings

**5 Takeaways:**

1. **Zero-shot ≠ Unbeatable**
   GPT-4o wins, but only by 5%

2. **Architecture Matters**
   ResNet blocks: +5-8% accuracy

3. **Data Augmentation is Crucial**
   ColorJitter, RandomErasing: +3-5%

4. **Upscaling Doesn't Help GPT-4o**
   32×32 = 224×224 (both 96.75%)

5. **Trade-offs Exist**
   CNN: Free & fast | GPT-4o: Expensive but zero-shot

---

## SLIDE 19 (CONCLUSIONS)

**Title:** 🎯 Conclusions

**Summary Stats:**
| Achievement | Value |
|-------------|-------|
| Baseline → Improved | +21% accuracy |
| Gap closed | 25% → 5% |
| Gap reduction | 80% |

**Main Takeaways:**
✅ Custom CNNs CAN compete with massive models
✅ Systematic improvements work (+21%)
✅ Trade-offs matter (cost vs accuracy vs speed)
✅ Zero-shot is powerful but not unbeatable

---

## SLIDE 20 (FUTURE WORK)

**Title:** 🔮 Future Work

**If We Had 6 More Months:**

• **Model:** Vision Transformers (ViT), attention mechanisms
• **Experiments:** Other LLMs (Claude, Gemini), harder datasets
• **Efficiency:** Quantization, edge deployment
• **Interpretability:** Grad-CAM visualizations

---

## SLIDE 21 (REFERENCES)

**Title:** 📚 References

1. CIFAR-10 Dataset - Krizhevsky (2009)
   https://www.cs.toronto.edu/~kriz/cifar.html

2. GPT-4o Vision - OpenAI (2024)
   https://platform.openai.com/docs/models/gpt-4o

3. ResNet - He et al. (2016)
   https://arxiv.org/abs/1512.03385

4. PyTorch - pytorch.org

5. Our Code:
   github.com/Sushmit404/cifar10-gpt4o-vision-test

---

## SLIDE 22 (THANK YOU)

**Title:** Questions?

**Center Text:** Thank You!

**Key Numbers:**
• Baseline CNN: 71.55%
• Improved CNN: 92.10%
• GPT-4o: 96.75%
• Gap Closed: 80%

**GitHub:** github.com/Sushmit404/cifar10-gpt4o-vision-test

---

# TIMING GUIDE (8 minutes)

| Slides | Who | Minutes |
|--------|-----|---------|
| 1-4 (Intro, Data) | Either | 1.5 |
| 5-6 (Baseline CNN) | Friend | 1.5 |
| 7-9 (GPT-4o + Fun) | You | 1.5 |
| 10-14 (Improvements) | Either | 2.0 |
| 15-20 (Results, Conclusions) | Either | 1.5 |

**TOTAL: 8 minutes**

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

💬 "Fun fact: those 4,000 API calls cost about $40... but hey, it's for science!"

💬 "The GPU was NOT happy during those 43 minutes of training"

💬 "At 32×32 pixels, honestly even WE couldn't tell some cats from dogs"

💬 "We thought upscaling would help GPT-4o, but nope - same accuracy!"

💬 "1.8 trillion parameters vs 1.2 million... that's like comparing a library to a post-it note"


