# CS260 Final Project Plan
## Trained vs. Zero-Shot: Custom CNN vs. GPT-4o Vision

---

## 🎯 Project Overview

**Research Question:**
How does a trained custom CNN perform on CIFAR-10 compared to GPT-4o Vision, a general-purpose model that performs zero-shot classification without CIFAR-10 fine-tuning?

**Dataset:** CIFAR-10 (60,000 images, 10 classes)
- Training: 50,000 images (for CNN only)
- Evaluation: 2,000 stratified test images (200 per class)

**Technologies:**
- **CNN Training:** PyTorch/TensorFlow
- **GPT-4o Testing:** OpenAI API, PIL, NumPy
- **Analysis:** scikit-learn, pandas
- **Visualization:** matplotlib, seaborn

---

## 📊 Two-Track Approach

### **Track A: Custom CNN (Training & Evaluation)**
*[Your team member handles this]*
1. Build CNN architecture (PyTorch/Keras)
2. Train on 50,000 CIFAR-10 images
3. Evaluate on 2,000 test subset
4. Extract feature maps
5. Generate CNN confusion matrix

### **Track B: GPT-4o Vision (Zero-Shot Evaluation)**
*[Your focus - this codebase]*
1. ✅ Setup pipeline (DONE)
2. Create 2,000 image stratified subset
3. Run GPT-4o evaluation via OpenAI API
4. Collect predictions and metrics
5. Generate GPT-4o confusion matrix
6. Analyze failure cases

---

## 🚀 GPT-4o Track: Detailed Task List

### **Phase 1: Data Preparation**
- [ ] **Task 1.1:** Create stratified sampling function
  - Select 200 images per class (total 2,000)
  - Ensure balanced representation
  - Save indices for reproducibility
  
- [ ] **Task 1.2:** Validate subset
  - Verify 200 images per class
  - Check for data leakage
  - Document sample statistics

### **Phase 2: Pipeline Enhancement**
- [ ] **Task 2.1:** Batch processing implementation
  - Process images in batches
  - Add rate limiting (OpenAI limits)
  - Implement delay between requests
  
- [ ] **Task 2.2:** Progress tracking
  - Add progress bar (tqdm)
  - Save checkpoints every 100 images
  - Resume from checkpoint if interrupted
  
- [ ] **Task 2.3:** Error handling
  - Retry failed API calls (3 attempts)
  - Log errors to file
  - Handle rate limit exceptions

### **Phase 3: Evaluation Execution**
- [ ] **Task 3.1:** Cost estimation
  - Calculate expected cost (~$20)
  - Verify API key has credits
  
- [ ] **Task 3.2:** Run evaluation
  - Process all 2,000 images
  - Save raw predictions
  - Track API response times
  
- [ ] **Task 3.3:** Data collection
  - Store: image_id, true_label, predicted_label, confidence
  - Save to CSV for analysis
  - Backup results to GitHub

### **Phase 4: Metrics & Analysis**
- [ ] **Task 4.1:** Calculate metrics
  - Overall accuracy
  - Per-class precision/recall/F1
  - Confusion matrix
  
- [ ] **Task 4.2:** Statistical analysis
  - Confidence intervals
  - Class-wise error rates
  - Common misclassifications

### **Phase 5: Visualization**
- [ ] **Task 5.1:** Confusion matrix
  - Heatmap with seaborn
  - Annotate with percentages
  - Highlight main errors
  
- [ ] **Task 5.2:** Comparison charts
  - Side-by-side: CNN vs GPT-4o accuracy
  - Per-class accuracy comparison
  - Error distribution
  
- [ ] **Task 5.3:** Failure case analysis
  - Visualize top 10 failures
  - Show true vs predicted labels
  - Identify patterns

### **Phase 6: Deliverables**
- [ ] **Task 6.1:** Export results
  - JSON: Full predictions
  - CSV: Summary metrics
  - PNG: All visualizations
  
- [ ] **Task 6.2:** Documentation
  - Update README with results
  - Write analysis section
  - Document cost breakdown
  
- [ ] **Task 6.3:** Final comparison
  - Merge CNN and GPT-4o results
  - Create unified comparison charts
  - Write discussion section

---

## 💰 Budget Estimate

### **GPT-4o Vision Costs**
- **Images:** 2,000
- **Cost per image:** ~$0.01
- **Total estimated cost:** ~$20
- **Rate limit:** 500 requests/min (should handle)

### **Recommendations**
- Start with 100 images to test pipeline
- Verify costs before full run
- Monitor API usage dashboard

---

## 📈 Expected Outcomes

### **GPT-4o Strengths (Expected)**
- Good on common classes (airplane, ship, truck)
- Robust to variations
- Zero-shot generalization

### **GPT-4o Weaknesses (Expected)**
- Confusion between similar animals (cat/dog)
- Lower accuracy than trained CNN
- Potential for out-of-vocabulary responses

### **CNN Strengths (Expected)**
- Higher overall accuracy (~90-95%)
- Optimized for CIFAR-10 specific features
- Fast inference

### **CNN Weaknesses (Expected)**
- Overfitting to training distribution
- Poor generalization to new domains
- Requires labeled training data

---

## 🔗 References

1. CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
2. OpenAI API Docs: https://platform.openai.com/docs/models/gpt-4o
3. Krizhevsky, A. (2009). Learning Multiple Layers of Features from Tiny Images
4. GitHub Repository: https://github.com/Sushmit404/cifar10-gpt4o-vision-test

---

## 📅 Timeline Suggestion

| Week | Tasks | Deliverable |
|------|-------|-------------|
| Week 1 | Tasks 1.1-2.3 | Enhanced pipeline ready |
| Week 2 | Tasks 3.1-3.3 | Full evaluation complete |
| Week 3 | Tasks 4.1-5.3 | Analysis & visualizations |
| Week 4 | Tasks 6.1-6.3 | Final report & comparison |

---

## 🤝 Collaboration Points

### **With CNN Track:**
- Share the 2,000 image subset indices
- Coordinate on visualization style
- Combine results for final comparison

### **Code Sharing:**
- Push all changes to GitHub
- Tag releases for milestones
- Document API usage

---

## ✅ Current Status

**Completed:**
- ✅ Basic pipeline setup
- ✅ CIFAR-10 download functionality
- ✅ GPT-4o API integration
- ✅ 5-image test successful
- ✅ GitHub repository created

**Next Steps:**
1. Implement stratified sampling (Task 1.1)
2. Add batch processing (Task 2.1)
3. Test on 100 images
4. Run full 2,000 image evaluation

