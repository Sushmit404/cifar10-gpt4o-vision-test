import torch
import torch.nn as nn
from train_cnn_32 import CustomCNN

model = CustomCNN(input_size=32)
params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {params:,}")
print(f"Trainable parameters: {trainable:,}")

