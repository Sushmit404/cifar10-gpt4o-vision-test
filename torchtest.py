# To test the PyTorch version and the GPU availability

import torch
print("PyTorch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print("device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
