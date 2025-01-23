import sys

import torch
import torchaudio
import torchvision

def printVersion():
    print(f"Is Cuda Enable?: {torch.cuda.is_available()}")
    print(f"Cuda Version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"Torchaudio version: {torchaudio.__version__}")
    print(f"Python version: {sys.version}")
