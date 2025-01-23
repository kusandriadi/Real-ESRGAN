import time

import metrics.cpu_and_ram as cpu_and_ram
import metrics.gpu as gpu


def monitor_system_metrics():
    """Memulai monitoring penggunaan sistem selama proses."""
    cpu_usage = []
    ram_usage_percent = []
    ram_usage_mb = []
    gpu_usage_percent = []
    gpu_memory_usage_percent = []
    gpu_memory_usage_mb = []

    start_time = time.time()

    def collect_metrics():
        cpu_percent, ram_percent, ram_used_mb, ram_total_mb = cpu_and_ram.get_cpu_ram_usage()
        gpu_percent, gpu_mem_percent, gpu_mem_used_mb, gpu_mem_total_mb = gpu.get_gpu_usage()

        cpu_usage.append(cpu_percent)
        ram_usage_percent.append(ram_percent)
        ram_usage_mb.append(ram_used_mb)
        gpu_usage_percent.append(gpu_percent)
        gpu_memory_usage_percent.append(gpu_mem_percent)
        gpu_memory_usage_mb.append(gpu_mem_used_mb)

    return start_time, collect_metrics, {
        "cpu_usage": cpu_usage,
        "ram_usage_percent": ram_usage_percent,
        "ram_usage_mb": ram_usage_mb,
        "gpu_usage_percent": gpu_usage_percent,
        "gpu_memory_usage_percent": gpu_memory_usage_percent,
        "gpu_memory_usage_mb": gpu_memory_usage_mb
    }

def calculate_average_metrics(metrics):
    """Menghitung rata-rata metrik yang dikumpulkan."""
    return {
        "avg_cpu": sum(metrics["cpu_usage"]) / len(metrics["cpu_usage"]) if metrics["cpu_usage"] else 0,
        "avg_ram_percent": sum(metrics["ram_usage_percent"]) / len(metrics["ram_usage_percent"]) if metrics["ram_usage_percent"] else 0,
        "avg_ram_used_mb": sum(metrics["ram_usage_mb"]) / len(metrics["ram_usage_mb"]) if metrics["ram_usage_mb"] else 0,
        "avg_gpu_percent": sum(metrics["gpu_usage_percent"]) / len(metrics["gpu_usage_percent"]) if metrics["gpu_usage_percent"] else 0,
        "avg_gpu_memory_percent": sum(metrics["gpu_memory_usage_percent"]) / len(metrics["gpu_memory_usage_percent"]) if metrics["gpu_memory_usage_percent"] else 0,
        "avg_gpu_memory_used_mb": sum(metrics["gpu_memory_usage_mb"]) / len(metrics["gpu_memory_usage_mb"]) if metrics["gpu_memory_usage_mb"] else 0,
    }