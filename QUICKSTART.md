# 🚀 Quick Start Guide

## One-Sentence Summary
Test GPT-4o Vision's image classification on CIFAR-10 by downloading the dataset (torchvision), converting images to PNG (PIL), calling OpenAI API (openai client), and comparing predictions vs ground truth (matplotlib).

---

## ⚡ 5 Simple Steps

### 1️⃣ **Install Dependencies (PyTorch, OpenAI, PIL)**
All packages are already installed via `requirements.txt`:
- `torch` & `torchvision` for CIFAR-10 dataset
- `openai` for GPT-4o Vision API
- `pillow` for image conversion
- `matplotlib` & `numpy` for visualization

### 2️⃣ **Get OpenAI API Key**
1. Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create account → "Create new secret key" → Copy it
3. Open `.env` file and replace `your_api_key_here` with your actual key

### 3️⃣ **Run the Pipeline (Main Script)**
```bash
python cifar10_gpt4o_test.py
```
This will:
- Download CIFAR-10 automatically (torchvision)
- Test 5 sample images with GPT-4o Vision (openai)
- Show accuracy and save visualization (matplotlib)

### 4️⃣ **View Results (PNG Output)**
Check `cifar10_gpt4o_results.png` for:
- ✅ Green titles = Correct predictions
- ❌ Red titles = Wrong predictions

### 5️⃣ **Scale to 2,000 Images (Optional)**
Edit line ~160 in `cifar10_gpt4o_test.py`:
```python
sample_indices = list(range(2000))  # Test first 2,000 images
```
⚠️ Cost: ~$20 for 2,000 images

---

## 🧪 Quiz: Test Your Understanding

**Q1**: Which library downloads the CIFAR-10 dataset?
<details>
<summary>Answer</summary>
`torchvision` - specifically `torchvision.datasets.CIFAR10`
</details>

**Q2**: Why do we upscale images to 224×224 pixels using PIL?
<details>
<summary>Answer</summary>
GPT-4o Vision performs better on larger images; CIFAR-10 images are only 32×32 pixels natively.
</details>

**Q3**: What format does the OpenAI API expect for images?
<details>
<summary>Answer</summary>
Base64-encoded PNG/JPEG - we convert PyTorch tensors to PNG bytes then encode to base64.
</details>

**Q4**: How do we prevent committing the API key to Git?
<details>
<summary>Answer</summary>
The `.env` file is listed in `.gitignore`, keeping secrets out of version control.
</details>

**Q5**: What technologies handle image visualization in the pipeline?
<details>
<summary>Answer</summary>
`matplotlib` for plotting and saving comparison images as PNG files.
</details>

---

## 📁 File Structure

```
CS260 Final Project/
├── cifar10_gpt4o_test.py    ← Main script (run this!)
├── requirements.txt          ← Python dependencies
├── .env                      ← Your API key (edit this!)
├── README.md                 ← Full documentation
├── QUICKSTART.md            ← This guide
└── data/                     ← Auto-created when script runs
```

---

## 🆘 Common Issues

### "OPENAI_API_KEY not found"
→ Edit `.env` file and add your real API key (no quotes needed)

### "Rate limit exceeded"
→ OpenAI limits requests per minute. Add delays or reduce sample size.

### "Module not found"
→ Run `pip install -r requirements.txt` again

---

## 🎯 Next Actions

1. ✅ Add your API key to `.env`
2. ✅ Run `python cifar10_gpt4o_test.py`
3. ✅ Check the output PNG
4. ✅ Experiment with different images
5. ✅ Analyze which classes GPT-4o struggles with

**Ready to scale?** Modify `sample_indices` to test hundreds or thousands of images!


