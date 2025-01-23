import psutil

def get_cpu_ram_usage():
    """Mengambil informasi penggunaan CPU dan RAM."""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    ram_info = psutil.virtual_memory()
    ram_percent = ram_info.percent
    ram_used_mb = ram_info.used / (1024 * 1024)
    ram_total_mb = ram_info.total / (1024 * 1024)
    return cpu_percent, ram_percent, ram_used_mb, ram_total_mb