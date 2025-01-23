import GPUtil

def get_gpu_usage():
    """Mengambil informasi penggunaan GPU."""
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_percent = sum(gpu.load for gpu in gpus) / len(gpus) * 100
        gpu_memory_used_mb = sum(gpu.memoryUsed for gpu in gpus)
        gpu_memory_total_mb = sum(gpu.memoryTotal for gpu in gpus)
        gpu_memory_percent = (gpu_memory_used_mb / gpu_memory_total_mb) * 100 if gpu_memory_total_mb > 0 else 0
        return gpu_percent, gpu_memory_percent, gpu_memory_used_mb, gpu_memory_total_mb
    return 0, 0, 0, 0