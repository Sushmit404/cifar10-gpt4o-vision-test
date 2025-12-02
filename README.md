# CIFAR-10 Classification with GPT-4o Vision
## CS260 Final Project

Test GPT-4o Vision's ability to classify CIFAR-10 images by comparing its predictions against ground truth labels.

---

## 🎯 Project Overview

This project evaluates GPT-4o Vision's performance on CIFAR-10 image classification by:
- **Dataset Handling** - Downloading CIFAR-10 using `torchvision`
- **Image Processing** - Converting PyTorch tensors to PNG format with `PIL`
- **API Integration** - Sending images to GPT-4o via `openai` Python client
- **Analysis** - Comparing predictions vs ground truth with visualization using `matplotlib`
- **Scalability** - Pipeline designed to handle up to 2,000+ images

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `openai` - OpenAI API client
- `torch` & `torchvision` - CIFAR-10 dataset loading
- `pillow` - Image processing
- `numpy` & `matplotlib` - Visualization
- `python-dotenv` - Environment variable management

### 2. Get Your OpenAI API Key

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Log in or create an account
3. Click **"Create new secret key"**
4. Copy the key

### 3. Configure API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_actual_api_key_here
```

⚠️ **Important**: Never commit your `.env` file to Git! It's already in `.gitignore`.

### 4. Run the Pipeline

```bash
python cifar10_gpt4o_test.py
```

---

## 📊 What the Script Does

1. **Downloads CIFAR-10** - Automatically downloads the test dataset (~170 MB)
2. **Selects Images** - Picks sample images (default: indices 0-4)
3. **Converts Format** - Transforms PyTorch tensors → PNG bytes (upscaled to 224×224)
4. **Calls GPT-4o Vision** - Sends images to OpenAI API for classification
5. **Compares Results** - Shows prediction vs ground truth
6. **Visualizes** - Saves comparison image as `cifar10_gpt4o_results.png`

---

## 🎨 Output Example

```
🚀 Starting CIFAR-10 + GPT-4o Vision Pipeline

📦 Downloading CIFAR-10 dataset...
✅ Downloaded! Test dataset size: 10000

🖼️  Selected 5 images
Ground-truth labels: ['cat', 'ship', 'ship', 'airplane', 'frog']

✅ OpenAI API key loaded

🔍 Testing images with GPT-4o Vision...

Processing image 0... ✅ True: cat | Predicted: cat
Processing image 1... ✅ True: ship | Predicted: ship
Processing image 2... ❌ True: ship | Predicted: boat
Processing image 3... ✅ True: airplane | Predicted: airplane
Processing image 4... ✅ True: frog | Predicted: frog

📈 Accuracy: 4/5 (80.0%)

🎨 Creating visualization...
📊 Results saved to cifar10_gpt4o_results.png

🎉 Pipeline complete!
```

---

## 🔧 Customization

### Test Different Images

Edit the `sample_indices` in `cifar10_gpt4o_test.py`:

```python
# Line ~160
sample_indices = [0, 1, 2, 3, 4]  # Change these!
```

### Scale to 2,000 Images

To test more images, modify:

```python
import random
sample_indices = random.sample(range(10000), 2000)  # Random 2000 images
```

⚠️ **Cost Warning**: GPT-4o Vision costs ~$0.01 per image. Testing 2,000 images ≈ $20.

---

## 📂 Project Structure

```
CS260 Final Project/
├── cifar10_gpt4o_test.py    # Main pipeline script
├── requirements.txt          # Python dependencies
├── .env                      # Your API key (create this!)
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── data/                    # CIFAR-10 dataset (auto-downloaded)
└── cifar10_gpt4o_results.png # Visualization output
```

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
- Make sure `.env` exists in the project root
- Check that it contains `OPENAI_API_KEY=your_key`
- Restart your terminal/IDE after creating `.env`

### "Rate limit exceeded"
- OpenAI has rate limits (e.g., 500 requests/min)
- Add a delay between requests: `time.sleep(0.5)` after each API call

### "Invalid API key"
- Verify your key at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- Make sure there are no extra spaces in `.env`

---

## 📝 CIFAR-10 Classes

The dataset contains 10 classes:
1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

---

## 🎓 Next Steps

- **Analyze Results**: Which classes does GPT-4o struggle with?
- **Compare Models**: Test GPT-4o-mini vs GPT-4o
- **Batch Processing**: Implement concurrent API calls for faster processing
- **Prompt Engineering**: Try different prompts to improve accuracy
- **Error Analysis**: Visualize misclassified images

---

## 📚 Technologies Used

- **Python 3.8+**
- **PyTorch & torchvision** - Dataset loading
- **OpenAI API** - GPT-4o Vision
- **PIL (Pillow)** - Image processing
- **NumPy** - Array operations
- **Matplotlib** - Visualization

---

## ⚖️ License

For educational use (CS260 Final Project).

---

**Questions?** Check the code comments or modify the script to experiment!


