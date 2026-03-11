import torch


print(f"torch version: {torch.__version__}")

use_gpu = False  # 想用CPU就 False
device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")