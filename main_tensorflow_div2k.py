import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from PIL import Image

from RealESRGAN import RealESRGAN

weight = 'x2'
scale = 2

def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights(f'weights/RealESRGAN_{weight}.pth', download=True)

    # Load the DIV2K dataset from TensorFlow Datasets
    dataset, info = tfds.load('div2k', split='train', with_info=True, as_supervised=True)

    for i, image in enumerate(tfds.as_numpy(dataset)):
        print(f'Processing image {i + 1}/{info.splits["train"].num_examples}')
        lr_image = tf_to_pil(lr_image)
        sr_image = model.predict(lr_image)

        image_name = f'DIV2K_{i + 1}'
        sr_image.save(f'results/{image_name}_result_{weight}_{scale}.png')


def tf_to_pil(image_tensor):
    # Convert TensorFlow tensor to PIL Image
    image = image_tensor.numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

if __name__ == '__main__':
    main()