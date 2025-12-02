# Trained vs. Zero-Shot: Custom CNN vs. GPT-4o Vision on CIFAR-10

A comprehensive comparison between a custom-trained Convolutional Neural Network (CNN) and OpenAI's GPT-4o Vision model for image classification on the CIFAR-10 dataset.

## 🎯 Project Overview

This project implements and compares two fundamentally different approaches to image classification:

1. **Custom CNN (Trained)**: A PyTorch-based convolutional neural network trained from scratch on 50,000 CIFAR-10 training images
2. **GPT-4o Vision (Zero-Shot)**: OpenAI's large multimodal model performing zero-shot classification without any CIFAR-10 specific training

Both models are evaluated on the same stratified subset of 2,000 test images (200 per class) to ensure fair comparison while keeping API costs manageable.

## 📊 Dataset

**CIFAR-10** contains 60,000 color images (32×32 pixels) across 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Data Split**:
- Training: 50,000 images (used only by CNN)
- Test subset: 2,000 images (stratified, 200 per class)
- Evaluation: Same 2,000-image subset for both models

## 🏗️ CNN Architecture

```
CustomCNN(
  (conv1): Conv2d(3, 32, kernel_size=3, padding=1)
  (bn1): BatchNorm2d(32)
  (pool1): MaxPool2d(2, 2)
  
  (conv2): Conv2d(32, 64, kernel_size=3, padding=1)
  (bn2): BatchNorm2d(64)
  (pool2): MaxPool2d(2, 2)
  
  (fc1): Linear(4096, 128)
  (dropout): Dropout(0.5)
  (fc2): Linear(128, 10)
)
```

**Features**:
- 2 convolutional layers with ReLU activation and max pooling
- Batch normalization for stable training
- Fully connected classification head with dropout
- Cross-entropy loss with Adam optimizer
- Data augmentation (random flip, random crop)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU (recommended for training, but CPU works)
- CUDA toolkit (if using GPU)
- OpenAI API key (for GPT-4o evaluation)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cifar10-gpt4o-vision-test.git
cd cifar10-gpt4o-vision-test
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up OpenAI API key**
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_api_key_here
```

## 📝 Usage

### Step 1: Train the CNN

```bash
python train_cnn.py
```

This will:
- Download CIFAR-10 dataset (if not already present)
- Train the CNN for 20 epochs (~10-30 minutes on GPU)
- Save the best model to `best_cnn_model.pth`
- Generate training history plots and metrics

**Expected training time**:
- GPU (NVIDIA RTX 3090): ~10 minutes
- GPU (NVIDIA GTX 1060): ~20-30 minutes
- CPU: ~2-3 hours

### Step 2: Compare Models

```bash
python compare_models.py
```

This will:
1. Create a stratified test subset (2,000 images, 200 per class)
2. Evaluate the trained CNN on the test subset
3. Evaluate GPT-4o Vision on the same test subset (requires API key)
4. Generate comprehensive comparison visualizations and metrics

**GPT-4o evaluation notes**:
- Makes 2,000 API calls (~30-60 minutes)
- Estimated cost: $10-20 USD
- Progress is saved every 50 images (resumable if interrupted)
- You'll be prompted to confirm before starting

### Step 3: View Results

All results are saved as PNG images and text files:
- `training_history.png` - Training and validation curves
- `cnn_confusion_matrix.png` - CNN prediction confusion matrix
- `gpt4o_confusion_matrix.png` - GPT-4o prediction confusion matrix
- `accuracy_comparison.png` - Overall and per-class accuracy comparison
- `cnn_feature_maps.png` - Visualization of learned CNN features
- `failure_cases.png` - Interesting cases where models succeed/fail
- `comparison_report.txt` - Detailed metrics and classification reports

## 📈 Expected Results

Based on typical performance:

| Model | Expected Accuracy | Training Time | Inference Time (2000 images) |
|-------|------------------|---------------|------------------------------|
| CNN (Trained) | 70-75% | 10-30 min (GPU) | < 1 minute |
| GPT-4o (Zero-Shot) | 85-95% | N/A | 30-60 minutes |

**Key Findings**:
- GPT-4o generally achieves higher accuracy due to massive pretraining
- CNN is much faster at inference time
- CNN performance is limited by small training dataset and simple architecture
- GPT-4o excels at fine-grained distinctions (e.g., cat vs dog)
- CNN may struggle with classes that require semantic understanding

## 📊 Metrics Reported

1. **Overall Accuracy**: Percentage of correct predictions
2. **Per-Class Accuracy**: Accuracy breakdown for each of 10 classes
3. **Precision, Recall, F1-Score**: Standard classification metrics
4. **Confusion Matrices**: Visual representation of prediction patterns
5. **Feature Maps**: Visualization of CNN learned features
6. **Failure Analysis**: Examples where each model succeeds/fails

## 🔧 Configuration

### Training Hyperparameters (in `train_cnn.py`)

```python
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
```

### Test Subset Size (in `compare_models.py`)

```python
num_samples = 2000  # Total test images (200 per class)
```

Adjust these if needed, but note:
- Larger batch size requires more GPU memory
- More epochs may improve accuracy but takes longer
- Larger test subset increases API costs for GPT-4o evaluation

## 💾 File Structure

```
cifar10-gpt4o-vision-test/
├── train_cnn.py              # Train the custom CNN
├── compare_models.py          # Compare CNN vs GPT-4o
├── cifar10_gpt4o_test.py     # Original GPT-4o test script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .env                       # OpenAI API key (create this)
├── data/                      # CIFAR-10 dataset (auto-downloaded)
├── best_cnn_model.pth        # Trained CNN model (generated)
├── training_history.json      # Training metrics (generated)
├── stratified_test_indices.json  # Test subset indices (generated)
├── gpt4o_progress.json       # GPT-4o evaluation progress (generated)
└── *.png, *.txt              # Results and visualizations (generated)
```

## 🎓 Discussion: Trained vs Zero-Shot

### CNN Strengths
- ✅ Fast inference (< 1ms per image)
- ✅ Deterministic and reproducible
- ✅ Can be fine-tuned for specific domains
- ✅ No API costs or internet dependency
- ✅ Complete control over architecture and training

### CNN Weaknesses
- ❌ Requires labeled training data
- ❌ Limited by training dataset size and quality
- ❌ Poor generalization to novel classes
- ❌ Training time and computational resources needed

### GPT-4o Vision Strengths
- ✅ Superior accuracy from massive pretraining
- ✅ Zero-shot capability (no training needed)
- ✅ Strong semantic understanding
- ✅ Can leverage world knowledge
- ✅ Generalizes well to novel concepts

### GPT-4o Vision Weaknesses
- ❌ Slow inference (~1-2 seconds per image)
- ❌ API costs (~$0.005-0.01 per image)
- ❌ Requires internet connection
- ❌ Black box (no interpretability)
- ❌ Non-deterministic outputs

## 🤝 Contributing

Feel free to open issues or submit pull requests for:
- Architecture improvements
- Better hyperparameter tuning
- Additional visualizations
- Extended comparisons (other models, datasets)

## 📄 License

MIT License - feel free to use this code for research or educational purposes.

## 🙏 Acknowledgments

- CIFAR-10 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/cifar.html), Alex Krizhevsky, 2009
- PyTorch framework: [PyTorch](https://pytorch.org/)
- OpenAI GPT-4o Vision: [OpenAI API](https://platform.openai.com/)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Happy experimenting! 🚀**
