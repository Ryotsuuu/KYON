import subprocess
import json
import shutil

def run_powershell(command):
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

print("=" * 80)
print("GPU HARDWARE INFORMATION")
print("=" * 80)

# Get Windows GPU information
ps_command = r'''
Get-CimInstance Win32_VideoController |
Select-Object Name, AdapterRAM, DriverVersion, DriverDate, PNPDeviceID, Status |
ConvertTo-Json -Depth 3
'''

output = run_powershell(ps_command)

if not output:
    print("Could not detect GPU information.")
    exit()

gpus = json.loads(output)

if isinstance(gpus, dict):
    gpus = [gpus]

for i, gpu in enumerate(gpus):

    print(f"\n{'=' * 80}")
    print(f"GPU {i}")
    print(f"{'=' * 80}")

    print(f"Name          : {gpu.get('Name')}")
    
    vram = gpu.get("AdapterRAM")
    if vram:
        print(f"VRAM          : {vram / (1024**3):.2f} GB")
    else:
        print("VRAM          : Unknown")

    print(f"Driver        : {gpu.get('DriverVersion')}")
    print(f"Driver Date   : {gpu.get('DriverDate')}")
    print(f"Status        : {gpu.get('Status')}")
    print(f"PCI/Device ID : {gpu.get('PNPDeviceID')}")

print("\n" + "=" * 80)
print("NVIDIA-SMI INFORMATION")
print("=" * 80)

if shutil.which("nvidia-smi"):

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,"
            "driver_version,pci.bus_id,compute_cap",
            "--format=csv"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)

else:
    print("nvidia-smi not found.")
    print("This usually means NVIDIA drivers/tools are not installed or")
    print("your laptop does not have an NVIDIA GPU.")

print("=" * 80)
