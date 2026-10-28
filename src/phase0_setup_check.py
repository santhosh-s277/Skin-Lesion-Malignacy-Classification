# SafeDerm Triage Lite - Phase 0 Setup & Environment Check
# Educational/research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.

import os
import sys
import platform
import subprocess

def check_system():
    divider = "=" * 70
    print(divider)
    print("SAFEDERM TRIAGE LITE - PHASE 0 SETUP CHECK")
    print("Disclaimer: Research prototype. Not a medical device. Not a diagnosis. Consult a qualified clinician.")
    print(divider)
    
    # 1. Python Environment Check
    py_ver = sys.version.split()[0]
    print(f"[1] Python Version: {py_ver} on {platform.system()} ({platform.architecture()[0]})")
    print(f"    Executable: {sys.executable}")
    is_py311 = sys.version_info.major == 3 and sys.version_info.minor == 11
    if is_py311:
        print("    Status: OK (Python 3.11 detected, compatible with PyTorch CUDA builds)")
    else:
        print(f"    Status: WARNING (Python 3.11 recommended, got {py_ver})")
        
    # 2. CUDA & GPU Check
    print("\n[2] Hardware & CUDA Diagnostics:")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"    PyTorch version: {torch.__version__}")
        print(f"    CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"    Total VRAM: {round(props.total_memory / (1024**3), 2)} GB")
        else:
            print("    WARNING: torch.cuda.is_available() is False")
    except ImportError:
        print("    PyTorch is not yet installed in this venv.")
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=gpu_name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True
            )
            print(f"    Host GPU (nvidia-smi): {res.stdout.strip()}")
        except Exception as e:
            print(f"    nvidia-smi error: {e}")
            
    # 3. File Inventory Check
    base_dir = r"C:\SafeDerm"
    print(f"\n[3] Project Files Status in {base_dir}:")
    
    files_to_check = {
        "Checkpoint": os.path.join(base_dir, r"checkpoints\best.pt"),
        "Model Code": os.path.join(base_dir, r"saved_model\model_code.py"),
        "Data Code": os.path.join(base_dir, r"data_code.py"),
        "Train Code": os.path.join(base_dir, r"train_code.py"),
        "Train Split": os.path.join(base_dir, r"splits\train.csv"),
        "Val Split": os.path.join(base_dir, r"splits\val.csv"),
        "Test Split": os.path.join(base_dir, r"splits\test.csv"),
        "Val+Test Tar": os.path.join(base_dir, r"ISIC2019_valtest_cache256.tar"),
        "Extracted Cache": os.path.join(base_dir, r"ISIC2019_cache256"),
        "Baseline Test Results": os.path.join(base_dir, r"results\test_predictions.csv"),
    }
    
    for name, path in files_to_check.items():
        if os.path.exists(path):
            if os.path.isfile(path):
                sz_mb = round(os.path.getsize(path) / (1024**2), 2)
                print(f"    [FOUND]     {name:<22}: {path} ({sz_mb} MB)")
            else:
                n_items = len(os.listdir(path))
                print(f"    [FOUND]     {name:<22}: {path} ({n_items} items)")
        else:
            print(f"    [MISSING]   {name:<22}: {path}")
            
    # 4. Check for External Downloads / Backups
    print("\n[4] External Archive Discovery:")
    backup_zip = os.path.expanduser("~") + r"\Downloads\ISIC2019_Important_Backup.zip"
    if os.path.exists(backup_zip):
        import zipfile
        with zipfile.ZipFile(backup_zip, "r") as z:
            names = z.namelist()
        print(f"    [FOUND] Backup Zip: {backup_zip} ({len(names)} files)")
        print(f"            - saved_model/model_code.py: {'saved_model/model_code.py' in names}")
        print(f"            - results/test_predictions.csv: {'results/test_predictions.csv' in names}")
        print(f"            - checkpoints/best.pt: {'checkpoints/best.pt' in names}")
    else:
        print(f"    [NOT FOUND] Backup Zip: {backup_zip}")
        
    local_whl_torch = r"D:\skinai_project\torch-2.5.1+cu121-cp311-cp311-win_amd64.whl"
    local_whl_vision = r"D:\skinai_project\torchvision-0.20.1+cu121-cp311-cp311-win_amd64.whl"
    if os.path.exists(local_whl_torch):
        print(f"    [FOUND] Offline PyTorch cu121 Wheel: {local_whl_torch}")
    if os.path.exists(local_whl_vision):
        print(f"    [FOUND] Offline TorchVision cu121 Wheel: {local_whl_vision}")

    print(divider)

if __name__ == "__main__":
    check_system()
