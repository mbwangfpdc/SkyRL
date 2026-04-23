import os
import sys
import torch

def print_section(title):
    print(f"\n{'='*10} {title} {'='*10}")

def debug_cuda():
    print_section("Environment Variables")
    # 1. Check what the OS is telling the process
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(f"CUDA_VISIBLE_DEVICES: {cvd if cvd is not None else 'Not Set (All visible)'}")
    
    print_section("PyTorch View")
    # 2. Check what PyTorch detects
    if not torch.cuda.is_available():
        print("❌ torch.cuda.is_available() returned FALSE")
        print("   -> Check your NVIDIA drivers or PyTorch installation.")
        return

    device_count = torch.cuda.device_count()
    print(f"CUDA Available: Yes")
    print(f"Device Count:   {device_count}")
    
    print_section("Device Connectivity Test")
    # 3. Try to actually touch every device
    for i in range(device_count):
        try:
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"✅ Device {i}: {name} ({mem:.2f} GB)")
            
            # 4. allocation test
            print(f"   Testing memory allocation on Device {i}...", end="", flush=True)
            x = torch.tensor([1.0], device=f"cuda:{i}")
            print(" OK")
            
        except Exception as e:
            print(f"\n❌ FAILED on Device {i}")
            print(f"   Error: {e}")

if __name__ == "__main__":
    debug_cuda()
