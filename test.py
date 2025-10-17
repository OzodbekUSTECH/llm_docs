import torch
print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

import onnxruntime as ort
print("ONNX провайдеры:", ort.get_available_providers())

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121