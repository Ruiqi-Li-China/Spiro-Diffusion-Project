import torch
import sys

print("--------------------------------------------------")
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")

# 检查 CUDA (NVIDIA显卡) 是否可用
if torch.cuda.is_available():
    print(f"CUDA available: YES (Device: {torch.cuda.get_device_name(0)})")
else:
    print("CUDA available: NO (Using CPU)")

print("--------------------------------------------------")
print("环境配置成功！你可以开始写代码了。")