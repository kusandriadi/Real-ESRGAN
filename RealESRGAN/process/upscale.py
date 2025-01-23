import os

import numpy as np
from PIL import Image

from metrics.psnr import calculate_psnr


def upscale(dataset_path, output_folder, dataset, model, collect_metrics, psnr_values):
    for scale_folder in os.listdir(dataset_path):
        scale_path = os.path.join(dataset_path, scale_folder)
        if not os.path.isdir(scale_path):
            continue

        for image_file in os.listdir(scale_path):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files

            image_name, _ = os.path.splitext(image_file)
            image_path = os.path.join(scale_path, image_file)
            image = Image.open(image_path).convert('RGB')

            # Load ground truth image from inputs/gt
            gt_path = os.path.join("inputs/gt", dataset, scale_folder, image_file)
            if not os.path.exists(gt_path):
                print(f"Ground truth file not found: {gt_path}")
                continue

            gt_image = Image.open(gt_path).convert('RGB')

            # Start monitoring before prediction
            collect_metrics()
            result_image = model.predict(image)

            # Determine output path based on folder structure
            result_dir = os.path.join(output_folder, dataset, scale_folder)
            os.makedirs(result_dir, exist_ok=True)

            result_path = os.path.join(result_dir, f"{image_name}.png")
            result_image.save(result_path)
            collect_metrics()

            # Calculate PSNR
            original_array = np.array(gt_image)
            result_array = np.array(result_image)

            if original_array.shape != result_array.shape:
                result_array = np.array(result_image.resize(original_array.shape[1::-1], Image.BICUBIC))

            psnr_values.append(calculate_psnr(original_array, result_array))

            print(f"Saved result for {image_file} to {result_path}")