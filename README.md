# Lab Notebook

## Progress Log and Time Estimates

Our initials are denoted to show who did what role. SC = Sushmit, AL = Alan. Time was relatively combined since we did most of this project when we were together.

### Week of November 7: Project Proposal. 

Some research on what to do or idea bouncing was mentioned in between, but we didn't do much before Thanksgiving break.

### December 1, 2025 (4 hours)

Initial Project Setup
- Initial commit: CIFAR-10 + GPT-4o testing pipeline (SC)
- Add comprehensive project plan for GPT-4o evaluation (SC)
- Start working on GPT API script (SC)
- Create stratified 2,000 image subset (SC)

### December 2, 2025 (3 hours)

- Discussion on project (SC)
- Started working on CNN code (AL)

### December 5, 2025 (2 hours)

- Finished working on CNN code (AL)
- Ran CNN for the first time (baseline model) (AL).
(Slightly inaccurate timing as I went to sleep when I let my CNN run).

### December 8, 2025 (3 hours)
- Started creating the slides for the final presentation (both)
- Coming up with ways to improve the CNN (AL).
- Ran GPT-4o analysis and gathered results (SC)
- After running GPT-4o analysis, research how to improve CNN (both) and start implementing updates (AL)

### December 9, 2025 (6 hours)
- Worked on improved CNN, ran CNN testing again using the same stratified subset. (AL)
-- This was a busy time, we had to meet up during lunch to work on improvements and I had to work on it during another class.
- Start Data Visualization (AL)
- Worked on slides for the final presentation

### December 10, 2025 (6-7 hours) 
- Finalized all data visualization (AL)
- Updating slides (both)
- Rehearsing presentation (both)

---

## References
“CIFAR-10 and CIFAR-100 datasets,” Toronto.edu, 2025. https://www.cs.toronto.edu/~kriz/cifar.html
Krizhevsky, A. Learning Multiple Layers of Features from Tiny Images. 2009.
“OpenAI Platform,” OpenAI.com, 2025. https://platform.openai.com/docs/models/gpt-4o
Both Alan and Sushmit worked on CNN research

---

## Technical Stack Used

Programming Language
- Python 3

Deep Learning Framework
- PyTorch (torch >= 2.0.0) - CNN model implementation, training, and evaluation
- torchvision (>= 0.15.0) - CIFAR-10 dataset loading, data transforms, and augmentation

Machine Learning Libraries
- scikit-learn (>= 1.3.0) - Stratified sampling and evaluation metrics
- scipy (>= 1.10.0) - Statistical analysis and curve fitting

API and Cloud Services
- OpenAI GPT-4o API - Zero-shot image classification
Sushmit spent $3 for this project as a result of API costs.

Data Processing and Image Handling
- NumPy (>= 1.24.0) - Numerical operations and array manipulation
- Pillow/PIL (>= 9.5.0) - Image processing and format conversion
- base64 - Image encoding for API transmission

Visualization
- Matplotlib (>= 3.7.0) - Confusion matrices, accuracy plots, and performance charts
- Seaborn (>= 0.12.0) - Enhanced statistical visualizations
- tqdm (>= 4.65.0) - Progress bars for training and evaluation loops

Dataset
- CIFAR-10 - 10-class image classification dataset (32x32 RGB images)
“CIFAR-10 and CIFAR-100 datasets,” Toronto.edu, 2025. https://www.cs.toronto.edu/~kriz/cifar.html

Alan's PC That They Ran Their CNN On (For Reference)
AMD Ryzen 7 9700X, 32GB DDR5 RAM, NVIDIA GeForce RTX 5070 Ti (16GB VRAM). 