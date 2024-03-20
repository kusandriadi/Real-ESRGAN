import os 
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import time
import psutil
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=8)
    model.load_weights('weights/RealESRGAN_x8.pth', download=True)

    with torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True) as prof:

        for i, image in enumerate(os.listdir("inputs")):
            image_name, _ = os.path.splitext(image)
            image = Image.open(f"inputs/{image}").convert('RGB')
            sr_image = model.predict(image)
            sr_image.save(f'results/{image_name}' + '_result_x8.png')

    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

if __name__ == '__main__':
    main()