"""
Given a PID, profiles CPU utilization for the process tree rooted at that PID.
"""
import pynvml
import numpy as np
import time
import csv
import threading
import sys
from datetime import datetime
from collections import defaultdict

shutdown_event = threading.Event()

data = defaultdict(list)
GPUS_TO_MONITOR = [1, 2, 3]

# Schema is (timestamp, mem_util, gpu_util)

def prof_gpus(output_dir):
    # return # Disabling this until we actually want to use pynvml
    pynvml.nvmlInit()
    try:
        while not shutdown_event.is_set():
            timestamp = datetime.now().strftime("%H:%M:%S")
            for i in GPUS_TO_MONITOR:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                # Note that reserved is provided as a "v2" of something or other
                data[i].append((timestamp, utilization.memory, utilization.gpu, mem_info.used, getattr(mem_info, "reserved", 0), mem_info.total))

            time.sleep(0.2)
    finally:
        pynvml.nvmlShutdown()
        for gpu_id, entries in data.items():
            with open(f"{output_dir}/gpu_{gpu_id}_usage_log.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ts", "mem_util", "gpu_util", "mem_used", "mem_reserved", "mem_total"])
                for entry in entries:
                    writer.writerow(entry)

if __name__ == "__main__":
    prof_gpus(sys.argv[1])
