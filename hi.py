import torch
print(f"Torch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

# 测试简单的张量运算，确保不会报 no kernel image
try:
    x = torch.randn(10, 10).cuda()
    y = torch.matmul(x, x)
    print("CUDA Test Passed! Tensor verification success.")
except Exception as e:
    print(f"CUDA Test Failed: {e}")
import torch
import vllm
print("Torch & vLLM imported successfully!")

import torch
import flash_attn
import torch
print(torch.cuda.get_device_capability())