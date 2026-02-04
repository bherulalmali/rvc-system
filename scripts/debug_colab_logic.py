import os
import shutil
import subprocess
import sys
from pathlib import Path

# Mocking Colab environment variables
PERSON_NAME = "test_voice_local"
EPOCHS = 1 # Quick test
AUDIO_FILES = ["dummy_audio.wav"] 

def run_local_test():
    print(f"🎤 Voice Name: {PERSON_NAME}")
    print(f"🔄 Epochs: {EPOCHS}")

    # 1. Setup Official RVC Backend
    RVC_BACKEND_DIR = "audio_processor_core"
    if not os.path.exists(RVC_BACKEND_DIR):
        print("📥 Cloning training backend...")
        subprocess.run(["git", "clone", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git", RVC_BACKEND_DIR], check=True)
    
    # ALWAYS ensure requirements are patched, even if directory exists (in case of re-runs)
    if os.path.exists(os.path.join(RVC_BACKEND_DIR, "requirements.txt")):
        print("🔧 Patching requirements.txt to avoid conflicts...")
        req_path = os.path.join(RVC_BACKEND_DIR, "requirements.txt")
        with open(req_path, "r") as f:
            lines = f.readlines()
        
        # Filter out problematic dependencies to install them manually later
        # aria2: fails on Mac, sometimes issues on Colab
        # fairseq: strictly pinned causing conflicts
        # faiss: explicitly handled later
        # numba/llvmlite: often conflict
        problematic = ["aria2", "fairseq", "faiss", "numba", "llvmlite"]
        new_lines = [line for line in lines if not any(p in line for p in problematic)]
        
        with open(req_path, "w") as f:
            f.writelines(new_lines)
            
    print("📦 Installing base requirements...")
    subprocess.run(f"cd {RVC_BACKEND_DIR} && {sys.executable} -m pip install -r requirements.txt", shell=True, check=True)
    
    print("🔧 Verifying training dependencies (simulating Colab fix)...")
    # Installing in the CURRENT environment (venv)
    # Re-adding stripped dependencies with safer constraints
    subprocess.run(f"{sys.executable} -m pip install faiss-cpu fairseq>=0.12.2 praat-parselmouth pyworld numba llvmlite --no-cache-dir", shell=True, check=True)
    subprocess.run(f"{sys.executable} -m pip install protobuf==3.20.0", shell=True, check=True)

    # 2. Prepare Dataset
    print("📂 Preparing dataset...")
    cwd_backup = os.getcwd()
    backend_abs_path = os.path.abspath(RVC_BACKEND_DIR)
    dataset_abs_path = os.path.join(backend_abs_path, "dataset", PERSON_NAME)
    
    if os.path.exists(dataset_abs_path):
        shutil.rmtree(dataset_abs_path)
    os.makedirs(dataset_abs_path)
    
    # Ensure logs folder exists
    logs_abs_path = os.path.join(backend_abs_path, "logs", PERSON_NAME)
    if not os.path.exists(logs_abs_path):
        os.makedirs(logs_abs_path)
    
    for audio_file in AUDIO_FILES:
        if os.path.exists(audio_file):
            shutil.copy(audio_file, dataset_abs_path)
        else:
            print(f"⚠️ File not found: {audio_file}")
            
    # 3. Trigger Training
    print("🧠 Starting Feature Extraction and Training...")
    
    os.chdir(RVC_BACKEND_DIR)
    try:
        def run_cmd(cmd):
            print(f"Running: {cmd}")
            # Capture output
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Command Failed!\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                raise RuntimeError(f"Command failed: {cmd}")
            print("✅ Done.")
            return result

        # Extract F0
        python_cmd = sys.executable 
        
        print("--- Testing Extract F0 ---")
        run_cmd(f"{python_cmd} infer/modules/train/extract/extract_f0_print.py '{dataset_abs_path}' 2 rmvpe")
        
        print("--- Testing Extract Features ---")
        # Force cpu for local test if no cuda
        run_cmd(f"{python_cmd} infer/modules/train/extract_feature_print.py cpu 1 0 0 '{dataset_abs_path}' v2")
        
        print("--- Testing Training (Dry Run) ---")
        cmd_train = f"{python_cmd} infer/modules/train/train.py -e {PERSON_NAME} -sr 40k -ov 0 -bs 4 -te {EPOCHS} -pg 0 -if 0 -l 0 -c 0 -sw 0 -v v2"
        run_cmd(cmd_train)
        
        print("✅ FULL SIMULATION SUCCESS! The code works.")
        
    except Exception as e:
        print(f"❌ Simulation failed with error: {e}")
    finally:
        os.chdir(cwd_backup)

if __name__ == "__main__":
    run_local_test()
