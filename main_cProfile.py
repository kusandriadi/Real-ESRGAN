import cProfile
import io
import os
import pstats

import torch
from PIL import Image

from RealESRGAN import RealESRGAN

weight = 'x2'
scale = 2

def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_{weight}.pth', download=True)

    for i, image in enumerate(os.listdir("inputs")):
        print(f'Gambar ke {i}')
        image_name, _ = os.path.splitext(image)
        image = Image.open(f"inputs/{image}").convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(f'results/{image_name}' + f'_result_{weight}_{scale}.png')

if __name__ == '__main__':
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()

    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())