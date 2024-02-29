import os
import subprocess
import sys
import torch
import logging
from configurations import CONFIGURATIONS  # Importing the configuration file

logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
                logging.FileHandler("run-hp.log"),
                logging.StreamHandler()])

def get_gpu_id(current_gpu_id, num_gpus):
    if num_gpus == 0:
        return -1
    return (current_gpu_id + 1) % num_gpus

def start_hp_tuner(num_instances, script_name='hp-tuner.py'):
    num_gpus = torch.cuda.device_count()
    current_gpu_id = -1

    current_gpu_id = get_gpu_id(current_gpu_id, num_gpus)
    period_years = CONFIGURATIONS.get('period_years', [1])[0]  # Default to [1] if not set
    args = [sys.executable, script_name, '--gpu_id', str(current_gpu_id),
            '--num_instances', str(num_instances), '--period_years', str(period_years)]
    subprocess.Popen(args)
    logging.info("Started instance")

if __name__ == "__main__":
    # Reading number of instances from environment variable or defaulting to 2
    num_instances = int(os.getenv('NUM_INSTANCES', 2))
    logging.info("Starting instance")
    start_hp_tuner(num_instances)
