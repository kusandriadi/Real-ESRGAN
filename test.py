import torch

if torch.cuda.is_available():
    print("Support GPU with Cuda")
    print("Cuda Version : " + torch.version.cuda)
    print("Pytorch Version : " + torch.version.__version__)
else:
    print("No GPU found; using CPU.")