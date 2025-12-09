# GPT-4o Vision Results Analysis
## CS260 Final Project - Ablation Study: Upscaling vs Raw Images

---

## 📊 Executive Summary

**Key Finding:** Upscaling CIFAR-10 images from 32×32 to 224×224 has **NO significant effect** on GPT-4o Vision's classification accuracy.

| Configuration | Accuracy | Correct/Total |
|--------------|----------|---------------|
| **224×224 (Upscaled)** | 96.75% | 1935/2000 |
| **32×32 (Raw)** | 96.75% | 1935/2000 |

Both achieve identical overall accuracy, but classify slightly different images correctly.

---

## 🔬 Detailed Comparison

### Overlap Analysis

| Category | Count | Percentage |
|----------|-------|------------|
| Both correct | 1915 | 95.75% |
| Both wrong | 45 | 2.25% |
| Only 224×224 correct | 20 | 1.00% |
| Only 32×32 correct | 20 | 1.00% |

### Interpretation
- **95.75%** of predictions are identical regardless of upscaling
- **2.25%** are consistently difficult (wrong in both)
- **2.00%** show resolution-dependent behavior (different outcomes)

---

## 📈 Per-Class Performance Comparison

| Class | 224×224 F1 | 32×32 F1 | Difference | Better Version |
|-------|------------|----------|------------|----------------|
| airplane | 99.0% | 98.5% | -0.5% | 224×224 |
| automobile | 97.0% | 97.5% | +0.5% | 32×32 |
| bird | 95.5% | 94.3% | -1.2% | 224×224 |
| cat | 92.8% | 92.6% | -0.2% | 224×224 |
| deer | 96.1% | 96.8% | +0.7% | 32×32 |
| dog | 94.9% | 95.2% | +0.3% | 32×32 |
| frog | 97.2% | 97.2% | 0.0% | Tie |
| horse | 98.5% | 98.8% | +0.3% | 32×32 |
| ship | 99.7% | 99.0% | -0.8% | 224×224 |
| truck | 96.7% | 98.0% | +1.3% | 32×32 |

### Observations
- **Vehicles (truck, automobile, deer)** tend to perform slightly better at raw 32×32
- **Flying objects (airplane, bird, ship)** tend to benefit from upscaling
- **Animals (cat, dog)** show mixed results with minimal difference
- **Frog** shows identical performance at both resolutions

---

## 🔍 Example Cases Where Resolution Mattered

### Cases Where Upscaling HELPED (224×224 ✅, 32×32 ❌)

| Image Index | True Label | 224×224 Prediction | 32×32 Prediction |
|-------------|------------|-------------------|------------------|
| 640 | dog | dog ✅ | cat ❌ |
| 3716 | cat | cat ✅ | bird ❌ |
| 1030 | cat | cat ✅ | dog ❌ |
| 4743 | frog | frog ✅ | bird ❌ |
| 9 | automobile | automobile ✅ | truck ❌ |

### Cases Where Upscaling HURT (224×224 ❌, 32×32 ✅)

| Image Index | True Label | 224×224 Prediction | 32×32 Prediction |
|-------------|------------|-------------------|------------------|
| 4864 | cat | dog ❌ | cat ✅ |
| 2181 | cat | dog ❌ | cat ✅ |
| 1559 | dog | cat ❌ | dog ✅ |
| 4251 | ship | airplane ❌ | ship ✅ |
| 923 | truck | automobile ❌ | truck ✅ |

---

## 📊 Confusion Patterns

### Common Misclassifications (Both Resolutions)
The 45 images that both versions got wrong reveal consistent confusion patterns:

1. **cat ↔ dog** - Most common confusion (fur texture similarity)
2. **automobile ↔ truck** - Vehicle shape overlap
3. **bird ↔ airplane** - Flying object confusion
4. **deer ↔ horse** - Four-legged animal similarity

---

## 💰 Cost Analysis

| Test | Images | Est. Cost | Time |
|------|--------|-----------|------|
| 224×224 Upscaled | 2,000 | ~$20 | ~42 min |
| 32×32 Raw | 2,000 | ~$20 | ~38 min |
| **Total** | 4,000 | ~$40 | ~80 min |

*Note: 32×32 was slightly faster due to smaller payload size.*

---

## 🎓 Conclusions for CS260 Paper

### Main Finding
> **Image resolution (32×32 vs 224×224) does not significantly impact GPT-4o Vision's classification accuracy on CIFAR-10. Both configurations achieve 96.75% accuracy.**

### Why This Matters
1. **Robustness:** GPT-4o Vision is resolution-agnostic for this task
2. **Efficiency:** No preprocessing (upscaling) is required
3. **Generalization:** The model handles extreme low-resolution images well
4. **Architecture:** Unlike CNNs with fixed input sizes, GPT-4o adapts dynamically

### Practical Implications
- For production use, skip upscaling to save computation
- For research, raw images provide valid results
- Resolution sensitivity is not a concern for GPT-4o Vision

---

## 📁 Result Files

| File | Description |
|------|-------------|
| `results_gpt4o/gpt4o_results_*.json` | 224×224 upscaled evaluation |
| `results_gpt4o_32x32/gpt4o_32x32_results_*.json` | 32×32 raw evaluation |
| `evaluate_gpt4o.py` | Upscaled evaluation script |
| `evaluate_gpt4o_no_upscale.py` | Raw 32×32 evaluation script |

---

## 📊 Comparison with CNN Baseline

| Model | Accuracy | Training Required | Cost |
|-------|----------|-------------------|------|
| **GPT-4o (224×224)** | 96.75% | None (zero-shot) | ~$20 |
| **GPT-4o (32×32)** | 96.75% | None (zero-shot) | ~$20 |
| **Custom CNN** | 64.75% | 50,000 images | Free |

**GPT-4o outperforms the trained CNN by 32 percentage points regardless of input resolution.**

---

## 📚 References

1. CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
2. OpenAI GPT-4o: https://platform.openai.com/docs/models/gpt-4o
3. Project Repository: https://github.com/Sushmit404/cifar10-gpt4o-vision-test

---

*Generated: December 9, 2025*
*CS260 Final Project - Trained vs. Zero-Shot Classification*

