import os

import torch
from PIL import Image
from torch.profiler import ProfilerActivity

from RealESRGAN import RealESRGAN

weight = 'x2'
scale = 2

def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_{weight}.pth', download=True)

    with torch.profiler.profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True) as prof:

        for i, image in enumerate(os.listdir("inputs")):
            print(f'Gambar ke {i}')
            image_name, _ = os.path.splitext(image)
            image = Image.open(f"inputs/{image}").convert('RGB')
            sr_image = model.predict(image)
            sr_image.save(f'results/{image_name}' + f'_result_{weight}_{scale}.png')

    # Print profiling results
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

if __name__ == '__main__':
    main()