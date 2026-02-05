import os
import shutil
import subprocess
import sys
from pathlib import Path

# --- CONFIGURATION (replicating Colab inputs) ---
PERSON_NAME = "local_test_voice"
EPOCHS = 2  # Keep it short for local test
# Use a dummy audio file for testing
AUDIO_FILES = ["dummy_audio.wav"] 

def create_dummy_audio():
    if not os.path.exists("dummy_audio.wav"):
        # Create a silent wav file using ffmpeg if possible, or just a placeholder
        # For this test, we REALLY need a valid wav or the tools will crash.
        # Let's see if we can use python to make one
        try:
            import wave
            import struct
            print("🔊 Creating dummy silent wav file...")
            with wave.open("dummy_audio.wav", "w") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(44100)
                # 1 second of silence
                data = struct.pack('<h', 0) * 44100
                f.writeframes(data)
        except Exception as e:
            print(f"⚠️ Failed to create dummy wav: {e}")

def run_simulation():
    print(f"🎤 Voice Name: {PERSON_NAME}")
    print(f"🔄 Epochs: {EPOCHS}")
    
    create_dummy_audio()

    # 1. Setup Official RVC Backend (ROBUST CLONE)
    RVC_BACKEND_DIR = "audio_processor_core"
    
    # FORCE CLEANUP for test
    if os.path.exists(RVC_BACKEND_DIR):
        if not os.path.exists(os.path.join(RVC_BACKEND_DIR, "infer")):
             print("⚠️ Detected broken backend. Deleting...")
             shutil.rmtree(RVC_BACKEND_DIR)
             
    if not os.path.exists(RVC_BACKEND_DIR):
        print("📥 Cloning training backend...")
        subprocess.run(["git", "clone", "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git", RVC_BACKEND_DIR], check=True)
    else:
        print("✅ Backend directory exists")
    
    print("📦 Installing/Checking Dependencies (Local Mock)...")
    # We skip actual pip install to avoid breaking local env, but we list what we would do
    deps = [
        "librosa==0.9.1", 
        "fairseq", # relax constraint
        "faiss-cpu",
        "praat-parselmouth==0.4.3",
        "pyworld==0.3.4",
        "tensorboardX",
        "torchcrepe",
        "ffmpeg-python",
        "av",
        "scipy",
        "protobuf==3.20.0"
    ]
    # For local test, we assume user has env set up or we'd break it.
    # We'll just verify imports for critical ones used in this script
    
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
    
    # DEBUG: Check file existence
    print("🔍 Validating backend files...")
    target_script = "infer/modules/train/extract/extract_f0_print.py"
    if not os.path.exists(target_script):
        print(f"❌ CRITICAL: Script not found: {target_script}")
    
    try:
        def run_cmd(cmd):
            print(f"Running: {cmd}")
            # We use sys.executable to ensure we use the current python env
            cmd = cmd.replace("python ", f"{sys.executable} ")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ Command Failed!\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                raise RuntimeError(f"Command failed: {cmd}")
            print("✅ Done.")
            return result
        
        # --- PREPROCESSING START ---
        print("--- 1. Preprocessing Dataset ---")
        # Arguments: <dataset_dir> <sample_rate> <n_threads> <exp_dir> <noparallel> <per>
        # This creating the '1_16k_wavs' and '2333333' folders required by extract_f0
        # NOTE: We must pass logs_abs_path as the exp_dir!
        cmd_preprocess = f"python infer/modules/train/preprocess.py '{dataset_abs_path}' 40000 2 '{logs_abs_path}' False 3.0"
        run_cmd(cmd_preprocess)
        # --- PREPROCESSING END ---

        # Check if pre-processing worked
        wavs_16k = os.path.join(logs_abs_path, "1_16k_wavs")
        # Note: preprocess.py puts files in LOGS dir (specifically under '1_16k_wavs' inside Experiment Name folder usually?)
        # Let's verify WHERE preprocess puts things.
        # Actually RVC preprocess takes dataset_path and puts output in ./logs/ExperimentName/
        # But wait, we passed dataset_abs_path. 
        # The script infer/modules/train/preprocess.py takes:
        # sys.argv[1]: trainset_dir
        # sys.argv[2]: sr (40000)
        # sys.argv[3]: n_p (threads)
        # It calculates exp_dir from... wait.
        # Let's look at how extract_f0 is called: run_cmd(f"python ... '{logs_abs_path}' ...")
        # So we expect preprocess to populate logs_abs_path?
        # Let's see what happens.
        
        # Extract F0
        print("--- 2. Extracting Pitch (F0) ---")
        run_cmd(f"python infer/modules/train/extract/extract_f0_print.py '{logs_abs_path}' 2 rmvpe")
        
        # Extract Features
        print("--- 3. Extracting Features ---")
        # Using cpu for local test if cuda not available
        method = "cuda" if "cuda" in sys.argv else "cpu"
        run_cmd(f"python infer/modules/train/extract_feature_print.py {method} 1 0 0 '{logs_abs_path}' v2")
        
        # Train
        print("--- 4. Training Model (Dry Run) ---")
        # We perform a very short training just to see if it starts
        cmd_train = f"python infer/modules/train/train.py -e {PERSON_NAME} -sr 40k -ov 0 -bs 4 -te {EPOCHS} -pg 0 -if 0 -l 0 -c 0 -sw 0 -v v2"
        run_cmd(cmd_train)
        
        print("✅ FULL SIMULATION SUCCESS! The code works.")
        
    except Exception as e:
        print(f"❌ Simulation failed with error: {e}")
    finally:
        os.chdir(cwd_backup)

if __name__ == "__main__":
    run_simulation()
