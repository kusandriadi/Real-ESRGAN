import os
import re
import time

import numpy as np
import torch

import version
from RealESRGAN import RealESRGAN
from RealESRGAN.process.upscale import upscale
from metrics.computation import monitor_system_metrics, calculate_average_metrics


def main() -> int:
    version.printVersion()

    start_time = time.time()

    # Loop melalui semua file weight
    weight_files = [f for f in os.listdir('weights') if f.endswith('.pth') and not f.startswith('1')]

    for weight_file in weight_files:
        # Ekstrak scale dari nama file weight
        match = re.search(r'_x(\d+)', weight_file)
        if not match:
            continue  # Lewati jika tidak ada skala dalam nama file

        scale = int(match.group(1))
        output_folder = f"output/{weight_file.split('.')[0]}"
        os.makedirs(output_folder, exist_ok=True)

        print("=====================================================")
        print(f"Processing weight: {weight_file} with scale: {scale}")
        print("=====================================================")

        # Buat nama file log berdasarkan waktu saat ini dan parameter
        log_file_path = f"{output_folder}/log_{weight_file.split('.')[0]}.log"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=scale)
        model.load_weights(f'weights/{weight_file}', download=False)

        monitor_start_time, collect_metrics, metrics = monitor_system_metrics()

        psnr_values = []

        with open(log_file_path, "w") as log_file:
            # Loop melalui semua folder di dalam inputs/lr
            for dataset in os.listdir("inputs/lr"):
                if dataset.startswith("1"):
                    continue  # Abaikan folder yang namanya dimulai dengan '1'

                dataset_path = os.path.join("inputs/lr", dataset)
                if not os.path.isdir(dataset_path):
                    continue

                # Call the upscale function
                upscale(dataset_path, output_folder, dataset, model, collect_metrics, psnr_values)

            # Hitung rata-rata PSNR
            avg_psnr = np.mean(psnr_values) if psnr_values else float('nan')

            # Hitung rata-rata metrik setelah semua gambar diproses
            avg_metrics = calculate_average_metrics(metrics)

            # Hitung waktu total
            total_time = time.time() - start_time

            # Tulis hasil ke log file
            log_file.write("\nHitung Metriks Gambar:\n")
            log_file.write(f"PSNR: {avg_psnr:.2f}\n")
            log_file.write("\nRata-rata penggunaan sistem selama proses:\n")
            log_file.write(f"CPU: {avg_metrics['avg_cpu']:.2f}%\n")
            log_file.write(f"RAM: {avg_metrics['avg_ram_percent']:.2f}% ({avg_metrics['avg_ram_used_mb']:.2f} MB)\n")
            log_file.write(f"GPU: {avg_metrics['avg_gpu_percent']:.2f}%\n")
            log_file.write(f"Memori GPU: {avg_metrics['avg_gpu_memory_percent']:.2f}% ({avg_metrics['avg_gpu_memory_used_mb']:.2f} MB)\n")
            log_file.write(f"Waktu total: {total_time:.2f} detik\n")

        print("\nMetriks gambar adalah:")
        print(f"PSNR: {avg_psnr:.2f}")

        print("\nRata-rata penggunaan sistem selama proses:")
        print(f"CPU: {avg_metrics['avg_cpu']:.2f}%")
        print(f"RAM: {avg_metrics['avg_ram_percent']:.2f}% ({avg_metrics['avg_ram_used_mb']:.2f} MB)")
        print(f"GPU: {avg_metrics['avg_gpu_percent']:.2f}%")
        print(f"Memori GPU: {avg_metrics['avg_gpu_memory_percent']:.2f}% ({avg_metrics['avg_gpu_memory_used_mb']:.2f} MB)")
        print(f"Waktu total: {total_time:.2f} detik")

if __name__ == '__main__':
    main()
