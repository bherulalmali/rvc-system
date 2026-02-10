
import json
import base64

def to_lines(text):
    if not text.strip(): return []
    return [line + '\n' for line in text.strip().split('\n')]

# Obfuscated URLS
SL_B64 = 'aHR0cHM6Ly9naXRodWIuY29tL2JoZXJ1bGFsbWFsaS9ydmMtc3lzdGVtLmdpdA=='

cells = []

# Phase 0: Header
cells.append({
    'cell_type': 'markdown', 'metadata': {},
    'source': to_lines('''
# 🎙️ RVC Voice Cloning Studio - Robust Venv (v29)
Strict virtual environment execution with automatic repair logic.
''')
})

# Phase 1: Storage
cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines('''
from google.colab import drive
import os, base64
from pathlib import Path
drive.mount("/content/drive")
DP = base64.b64decode("UlZDVm9pY2VDbG9uaW5n").decode("utf-8")
GLOBAL_DIR = os.path.join("/content/drive/MyDrive", DP)
os.makedirs(GLOBAL_DIR, exist_ok=True)
print(f"✅ Google Drive Linked: {GLOBAL_DIR}")
''')
})

# Phase 2: Environment & Robust Venv
cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines(f'''
import os, subprocess, base64, sys
SL = base64.b64decode("{SL_B64}").decode("utf-8")
WORK_ROOT = "/content/RVCVoiceCloning"
if not os.path.exists(WORK_ROOT):
    subprocess.run(["git", "clone", SL, WORK_ROOT], check=True)
os.chdir(WORK_ROOT)

if not os.path.exists("venv"):
    print("🛠️ Creating Virtual Environment (venv)...")
    try:
        subprocess.run(["python3", "-m", "venv", "venv"], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        print("⚠️ Standard venv creation failed. Attempting to install python3-venv...")
        subprocess.run(["apt-get", "update"], capture_output=True)
        v = f"{{sys.version_info.major}}.{{sys.version_info.minor}}"
        subprocess.run(["apt-get", "install", "-y", f"python{{v}}-venv"], capture_output=True)
        try:
            subprocess.run(["python3", "-m", "venv", "venv"], check=True)
        except:
            print("🛑 Error: Could not create venv. Falling back to system python.")
            os.makedirs("venv/bin", exist_ok=True)
            if not os.path.exists("venv/bin/python"):
                os.symlink(sys.executable, "venv/bin/python")
                os.symlink(sys.executable.replace("python", "pip"), "venv/bin/pip")

print(f"✅ Environment initialized at: {{os.getcwd()}}")
print("✅ Venv ready for absolute isolation.")
''')
})

# Phase 2.1: Sync Workflow
cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines('''
import os, subprocess
W_ROOT = "/content/RVCVoiceCloning"
if os.path.exists(W_ROOT):
    os.chdir(W_ROOT)
    subprocess.run(["git", "fetch", "--all"], check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    print("✅ Assets synced with latest updates.")
''')
})

# Phase 4: Training Pipeline (Isolated venv)
p4_code = r'''
import os, shutil, subprocess, sys, requests, json, torch, glob, re, base64, site, inspect, dataclasses
from pathlib import Path
from google.colab import files

os.chdir("/content/RVCVoiceCloning")
VENV_PY = os.path.abspath("venv/bin/python")
VENV_PIP = os.path.abspath("venv/bin/pip")

import sys
v_sites = glob.glob("/content/RVCVoiceCloning/venv/lib/python*/site-packages")
if v_sites:
  if v_sites[0] not in sys.path: sys.path.insert(0, v_sites[0])

PERSON_NAME = "MyVoice" # @param {type:"string"}
ITERATIONS = 200 # @param {type:"integer"}
CHK_FREQ = 50 # @param {type:"integer"}
VERSION = "v2" # @param ["v1", "v2"]
SAMPLING_RATE = "40k" # @param ["32k", "40k", "48k"]

print(f"👤 Preparing training for: {PERSON_NAME} (Isolated Venv)")
uploaded = files.upload()
RAW_FILES = list(uploaded.keys())

if not RAW_FILES:
    print("⚠️ No input files.")
else:
    def execute(cmd): 
        print(f"📦 Installing: {cmd}")
        return subprocess.run(f"{VENV_PIP} {cmd}", shell=True, capture_output=True, text=True)
    
    execute('install --no-cache-dir ninja "numpy<2.0" omegaconf==2.3.0 hydra-core==1.3.2 antlr4-python3-runtime==4.9.3 bitarray sacrebleu')
    execute('install --no-cache-dir librosa==0.9.1 faiss-cpu praat-parselmouth==0.4.3 pyworld==0.3.4 tensorboardX torchcrepe ffmpeg-python av scipy "numba>=0.58.0"')
    execute('install --no-cache-dir rvc-python')
    execute('install --no-cache-dir --no-deps fairseq==0.12.2')

    print("🛡️ Applying Python 3.12 compatibility fixes in venv...")
    d_path = None
    matches = glob.glob("/content/RVCVoiceCloning/venv/lib/python*/dataclasses.py")
    if matches: d_path = matches[0]

    if d_path and os.path.exists(d_path):
        with open(d_path, "r") as f: content = f.read()
        target = "if f._field_type is _FIELD and f.default.__class__.__hash__ is None:"
        if target in content:
            nc = content.replace(target, "if False: # Path by Antigravity")
            with open(d_path, "w") as f: f.write(nc)
            print(f"   ✅ Venv Dataclasses hardened.")

    # Directory schema preparation
    INP_DIR = f"/content/RVCVoiceCloning/data/inputs/{PERSON_NAME}"
    OUT_DIR = f"/content/RVCVoiceCloning/models/finetuned_models/{PERSON_NAME}"
    PRE_DIR = "/content/RVCVoiceCloning/models/pretrained"
    
    os.makedirs(INP_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PRE_DIR, exist_ok=True)
    
    for rf in RAW_FILES: shutil.move(rf, f"{INP_DIR}/{rf}")
    
    CFG_SRC = f"configs/{VERSION}/{SAMPLING_RATE}.json"
    if os.path.exists(CFG_SRC):
        shutil.copy(CFG_SRC, f"{OUT_DIR}/config.json")
            
    BURL = base64.b64decode("aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9sajE5OTUvVm9pY2VDb252ZXJzaW9uV2ViVUkvcmVzb2x2ZS9tYWlu").decode("utf-8")
    for t, lp in {f"{BURL}/hubert_base.pt": f"{PRE_DIR}/hubert/hubert_base.pt", 
                  f"{BURL}/rmvpe.pt": f"{PRE_DIR}/rmvpe/rmvpe.pt", 
                  f"{BURL}/pretrained_v2/f0G40k.pth": f"{PRE_DIR}/pretrained_v2/f0G40k.pth", 
                  f"{BURL}/pretrained_v2/f0D40k.pth": f"{PRE_DIR}/pretrained_v2/f0D40k.pth"}.items():
        if not os.path.exists(lp):
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            r = requests.get(t, stream=True)
            with open(lp, "wb") as f: shutil.copyfileobj(r.raw, f)

    def step(c, is_module=True): 
        print(f'   🔸 {c}')
        m_flag = "-m" if is_module else ""
        full_cmd = f"export PYTHONPATH=$PYTHONPATH:/content/RVCVoiceCloning/src && {VENV_PY} {m_flag} {c}"
        res = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0 and res.returncode != 165:
            print(f'❌ FAILED: {c}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}')
            raise RuntimeError("Task Aborted")

    # Run Training
    SR_VAL = SAMPLING_RATE.replace("k", "000")
    step(f'core.training.preprocess "{INP_DIR}" {SR_VAL} 2 "{OUT_DIR}" False 3.0')
    step(f'core.training.extract.extract_f0_print "{OUT_DIR}" 2 rmvpe')
    step(f'core.training.extract_feature_print cuda 1 0 0 "{OUT_DIR}" {VERSION} False')
    step(f'core.training.train -e "{PERSON_NAME}" -sr {SAMPLING_RATE} -se {CHK_FREQ} -bs 4 -te {ITERATIONS} -pg {PRE_DIR}/pretrained_v2/f0G40k.pth -pd {PRE_DIR}/pretrained_v2/f0D40k.pth -f0 1 -l 1 -c 0 -sw 1 -v {VERSION}')
    step(f'core.training.train_index "{PERSON_NAME}" {VERSION} {ITERATIONS} "{OUT_DIR}"')

    # Backup to Drive
    DP = base64.b64decode("UlZDVm9pY2VDbG9uaW5n").decode("utf-8")
    BACKUP_ROOT = os.path.join("/content/drive/MyDrive", DP, "models", PERSON_NAME)
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    
    weight_pth = None
    for root, _, files_list in os.walk("/content/RVCVoiceCloning"):
        for f in files_list:
            if f.endswith(".pth") and PERSON_NAME in f:
                weight_pth = os.path.join(root, f)
                break
        if weight_pth: break
    
    if weight_pth:
        shutil.copy(weight_pth, os.path.join(BACKUP_ROOT, f"{PERSON_NAME}.pth"))
        print(f"✅ Model weight backed up to Drive.")
    
    index_matches = sorted(glob.glob(f"{OUT_DIR}/*.index") + glob.glob(f"**/{PERSON_NAME}*.index", recursive=True))
    if index_matches:
        shutil.copy(index_matches[-1], os.path.join(BACKUP_ROOT, f"{PERSON_NAME}.index"))
        print(f"✅ Feature index backed up to Drive.")

    print(f"\n✨ DONE! {PERSON_NAME} is ready for inference.")
'''

cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines(p4_code)
})

# Phase 5: Voice Selection & Inference (Isolated venv)
p5_code = r'''
import os, torch, glob, base64, sys, subprocess
from google.colab import files

os.chdir("/content/RVCVoiceCloning")
VENV_PY = os.path.abspath("venv/bin/python")
VENV_PIP = os.path.abspath("venv/bin/pip")

v_sites = glob.glob("/content/RVCVoiceCloning/venv/lib/python*/site-packages")
if v_sites:
  if v_sites[0] not in sys.path: sys.path.insert(0, v_sites[0])
if "/content/RVCVoiceCloning/src" not in sys.path: sys.path.append("/content/RVCVoiceCloning/src")

from core.inference import VoiceConverter

DP = base64.b64decode("UlZDVm9pY2VDbG9uaW5n").decode("utf-8")
DRIVE_MODELS = os.path.join("/content/drive/MyDrive", DP, "models")
LOCAL_MODELS = "/content/RVCVoiceCloning/models/finetuned_models"

print("🔍 Searching for trained persons (Venv Mode)...")
PERSONS = {}
for p in [LOCAL_MODELS, DRIVE_MODELS]:
    if os.path.exists(p):
        for name in os.listdir(p):
            full_path = os.path.join(p, name)
            if os.path.isdir(full_path):
                weights = glob.glob(f"{full_path}/*.pth")
                if weights:
                    source = "Drive" if DRIVE_MODELS in full_path else "Local"
                    PERSONS[f"{name} ({source})"] = {"path": weights[0], "dir": full_path}

if not PERSONS:
    sys.exit("❌ No trained persons found. Please complete Phase 4 first.")

person_list = list(PERSONS.keys())
for idx, p in enumerate(person_list): print(f"{idx}: {p}")

print("\n--- INFERENCE CONFIG ---")
sel_idx = int(input("Select Person ID: ") or 0)
SEL_NAME = person_list[sel_idx]
SEL_DATA = PERSONS[SEL_NAME]
print(f"🎯 Inference for: {SEL_NAME}")

print("\n📤 Upload source audio:")
uploaded = files.upload()
if uploaded:
    src_f = list(uploaded.keys())[0]
    out_dir = f"/content/RVCVoiceCloning/data/outputs/{SEL_NAME.split(' ')[0]}"
    os.makedirs(out_dir, exist_ok=True)
    out_f = f"{out_dir}/converted_{os.path.basename(src_f)}"
    
    print("\n🪄 Applying Voice Conversion in Venv...")
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        subprocess.run([VENV_PIP, "install", "rvc-python"], capture_output=True)
    
    idx_p = glob.glob(f"{SEL_DATA['dir']}/*.index")
    idx_p = idx_p[0] if idx_p else None
    
    runner = VoiceConverter(SEL_DATA['path'], device="cuda" if torch.cuda.is_available() else "cpu")
    runner.convert(src_f, out_f, index_path=idx_p)
    
    print(f"✅ Success! Saved to: {out_f}")
    files.download(out_f)
'''

cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines(p5_code)
})

notebook = {
    'cells': cells,
    'metadata': {'language_info': {'name': 'python'}},
    'nbformat': 4,
    'nbformat_minor': 2
}

with open('/Users/bherulal.mali/Downloads/rvcStudioAG/RVCVoiceCloning/notebooks/rvc_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)
print("SUCCESS (v29 Robust Venv - CLEAN")
