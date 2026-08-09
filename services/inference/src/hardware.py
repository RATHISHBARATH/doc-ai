import os
import subprocess

def detect_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'],
                                capture_output=True, text=True, check=True)
        total_vram = int(result.stdout.strip().split()[0])
        return total_vram > 0
    except:
        return False

def get_hardware_config():
    return {
        'gpu_available': detect_gpu(),
        'cpu_cores': os.cpu_count(),
    }
