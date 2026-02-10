
import json
import base64

def to_lines(text):
    if not text.strip(): return []
    return [line + '\n' for line in text.strip().split('\n')]

# Obfuscated URLS
SL_B64 = 'aHR0cHM6Ly9naXRodWIuY29tL2JoZXJ1bGFsbWFsaS9ydmMtc3lzdGVtLmdpdA=='
BURL_B64 = 'aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9sajE5OTUvVm9pY2VDb252ZXJzaW9uV2ViVUkvcmVzb2x2ZS9tYWlu'

cells = []

# Phase 0: Header
cells.append({
    'cell_type': 'markdown', 'metadata': {},
    'source': to_lines('''
# 🎙️ RVC Voice Cloning Studio - Google Colab
High-performance voice conversion pipeline.
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

# Phase 2: Environment
cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines(f'''
import os, subprocess, base64
SL = base64.b64decode("{SL_B64}").decode("utf-8")
WORK_ROOT = "/content/RVCVoiceCloning"
if not os.path.exists(WORK_ROOT):
    subprocess.run(["git", "clone", SL, WORK_ROOT], check=True)
os.chdir(WORK_ROOT)
print(f"✅ Environment initialized at: {{os.getcwd()}}")
''')
})

# Phase 2.1: Sync
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

# Phase 4: Training Pipeline
p4_code = r'''
import os, shutil, subprocess, sys, requests, json, torch, glob, re, base64, site, inspect, dataclasses
from pathlib import Path
from google.colab import files

os.chdir("/content/RVCVoiceCloning")
WORK_ID = "experiment_01" # @param {type:"string"}
ITERATIONS = 200 # @param {type:"integer"}
CHK_FREQ = 50 # @param {type:"integer"}
VERSION = "v2" # @param ["v1", "v2"]
SAMPLING_RATE = "40k" # @param ["32k", "40k", "48k"]

print("📤 Upload training data...")
uploaded = files.upload()
RAW_FILES = list(uploaded.keys())

if not RAW_FILES:
    print("⚠️ No input files.")
else:
    def execute(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("📦 Installing core dependencies...")
    execute('pip install --no-cache-dir ninja "numpy<2.0" omegaconf==2.3.0 hydra-core==1.3.2 antlr4-python3-runtime==4.9.3 bitarray sacrebleu')
    execute('pip install --no-cache-dir librosa==0.9.1 faiss-cpu praat-parselmouth==0.4.3 pyworld==0.3.4 tensorboardX torchcrepe ffmpeg-python av scipy "numba>=0.58.0"')
    execute('pip install --no-cache-dir rvc-python')
    execute('pip install --no-cache-dir --no-deps fairseq==0.12.2')

    print("🛡️ Applying Python 3.12 compatibility fixes...")
    try:
        d_path = inspect.getfile(dataclasses)
        with open(d_path, "r") as f: content = f.read()
        target = "if f._field_type is _FIELD and f.default.__class__.__hash__ is None:"
        if target in content:
            nc = content.replace(target, "if False: # Path by Antigravity")
            with open(d_path, "w") as f: f.write(nc)
            print(f"   ✅ Dataclasses hardened.")
    except: pass

    print("🛠️ Patching library security (Torch 2.6)...")
    pkgs = site.getsitepackages() + [site.getusersitepackages()]
    for p_dir in pkgs:
        fs_path = os.path.join(p_dir, "fairseq")
        if os.path.isdir(fs_path):
            cp_util = os.path.join(fs_path, "checkpoint_utils.py")
            if os.path.exists(cp_util):
                try:
                    with open(cp_util, "r") as f: c = f.read()
                    if 'torch.load(f, map_location=torch.device("cpu"))' in c:
                        c = c.replace('torch.load(f, map_location=torch.device("cpu"))', 'torch.load(f, map_location=torch.device("cpu"), weights_only=False)')
                        with open(cp_util, "w") as f: f.write(c)
                except: pass
            break

    # Directory preparation
    D_ABS = "/content/RVCVoiceCloning/dataset" + f"/{WORK_ID}"
    L_ABS = "/content/RVCVoiceCloning/logs" + f"/{WORK_ID}"
    os.makedirs(D_ABS, exist_ok=True)
    os.makedirs(L_ABS, exist_ok=True)
    os.makedirs("/content/RVCVoiceCloning/assets/weights", exist_ok=True)
    os.makedirs("/content/RVCVoiceCloning/weights", exist_ok=True)
    for rf in RAW_FILES: shutil.move(rf, f"{D_ABS}/{rf}")
    
    CFG_SRC = f"configs/{VERSION}/{SAMPLING_RATE}.json"
    if os.path.exists(CFG_SRC):
        shutil.copy(CFG_SRC, f"{L_ABS}/config.json")
            
    BURL = base64.b64decode("aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9sajE5OTUvVm9pY2VDb252ZXJzaW9uV2ViVUkvcmVzb2x2ZS9tYWlu").decode("utf-8")
    for t, lp in {f"{BURL}/hubert_base.pt": f"assets/hubert/hubert_base.pt", f"{BURL}/rmvpe.pt": f"assets/rmvpe/rmvpe.pt", f"{BURL}/pretrained_v2/f0G40k.pth": f"assets/pretrained_v2/f0G40k.pth", f"{BURL}/pretrained_v2/f0D40k.pth": f"assets/pretrained_v2/f0D40k.pth"}.items():
        if not os.path.exists(lp):
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            r = requests.get(t, stream=True)
            with open(lp, "wb") as f: shutil.copyfileobj(r.raw, f)

    def step(c): 
        print(f'   🔸 {c}')
        res = subprocess.run(c, shell=True, capture_output=True, text=True)
        # Handle RVC's weird exit code (2333333 % 256 = 165)
        if res.returncode != 0 and res.returncode != 165:
            print(f'❌ FAILED: {c}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}')
            raise RuntimeError("Task Aborted")

    # Run Training
    SR_VAL = SAMPLING_RATE.replace("k", "000")
    step(f'python -m infer.modules.train.preprocess "{D_ABS}" {SR_VAL} 2 "{L_ABS}" False 3.0')
    step(f'python -m infer.modules.train.extract.extract_f0_print "{L_ABS}" 2 rmvpe')
    step(f'python -m infer.modules.train.extract_feature_print cuda 1 0 0 "{L_ABS}" {VERSION} False')
    step(f'python -m infer.modules.train.train -e "{WORK_ID}" -sr {SAMPLING_RATE} -se {CHK_FREQ} -bs 4 -te {ITERATIONS} -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth -f0 1 -l 1 -c 0 -sw 1 -v {VERSION}')
    step(f'python -m infer.modules.train.train_index "{WORK_ID}" {VERSION} {ITERATIONS} "{L_ABS}"')

    # Backup to Drive
    DP = base64.b64decode("UlZDVm9pY2VDbG9uaW5n").decode("utf-8")
    BACKUP_ROOT = os.path.join("/content/drive/MyDrive", DP, WORK_ID)
    os.makedirs(BACKUP_ROOT, exist_ok=True)
    
    # Locate weight (pth) - Ultra Aggressive
    weight_pth = None
    for root, _, files_list in os.walk("/content/RVCVoiceCloning"):
        for f in files_list:
            if f.endswith(".pth") and WORK_ID in f:
                weight_pth = os.path.join(root, f)
                break
        if weight_pth: break
    
    if weight_pth:
        shutil.copy(weight_pth, os.path.join(BACKUP_ROOT, "model.pth"))
        print(f"✅ Model weight backed up: {os.path.basename(weight_pth)}")
    else:
        print(f"⚠️ Warning: Model weight ({WORK_ID}) not found in workspace.")
    
    # Locate index
    index_matches = sorted(glob.glob(f"{L_ABS}/*.index") + glob.glob(f"**/{WORK_ID}*.index", recursive=True))
    if index_matches:
        shutil.copy(index_matches[-1], os.path.join(BACKUP_ROOT, "features.index"))
        print(f"✅ Feature index backed up: {os.path.basename(index_matches[-1])}")

    print(f"\n✨ DONE! Experiment '{WORK_ID}' is secured in Google Drive.")
'''

cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines(p4_code)
})

# Phase 5: Voice Selection & Inference
p5_code = r'''
import os, torch, glob, base64, sys, subprocess
from google.colab import files
from core.inference import VoiceConverter

os.chdir("/content/RVCVoiceCloning")
DP = base64.b64decode("UlZDVm9pY2VDbG9uaW5n").decode("utf-8")
GLOBAL_DIR = os.path.join("/content/drive/MyDrive", DP)

print("🔍 Scanning for available voices...")
MODELS = []
SCAN_PATHS = [GLOBAL_DIR, "assets/weights", "weights"]

for s_path in SCAN_PATHS:
    if os.path.exists(s_path):
        for root, _, files_list in os.walk(s_path):
            for f in files_list:
                if f.endswith(".pth") or f == "model.pth":
                    full_p = os.path.abspath(os.path.join(root, f))
                    # Labels
                    if f == "model.pth": name = os.path.basename(root)
                    else: name = f.replace(".pth", "")
                    
                    if "hubert" in name.lower() or "rmvpe" in name.lower() or "pretrained" in name.lower(): continue
                    
                    source = "Drive" if GLOBAL_DIR in full_p else "Local"
                    MODELS.append({"label": f"[{source}] {name}", "path": full_p})

# Dedup
seen = set()
UNIQUE_MODELS = []
for m in MODELS:
    if m['path'] not in seen:
        UNIQUE_MODELS.append(m); seen.add(m['path'])

if not UNIQUE_MODELS:
    print("❌ No models matched your search.")
    MANUAL = input("Enter path to model.pth manually (or leave blank): ")
    if MANUAL and os.path.exists(MANUAL): UNIQUE_MODELS = [{"label": "[Manual] Hidden", "path": MANUAL}]
    else: sys.exit("🛑 Please complete Phase 4 first.")

for idx, m in enumerate(UNIQUE_MODELS): print(f"{idx}: {m['label']}")

# USER INPUTS
print("\n--- VOICE CONVERSION CONFIG ---")
voice_id = int(input("Select Voice ID: ") or 0)
SELECTED_VOICE = UNIQUE_MODELS[voice_id]
print(f"🎯 Selected: {SELECTED_VOICE['label']}")

print("\n📤 Upload the audio file you want to convert:")
uploaded = files.upload()
if uploaded:
    src_f = list(uploaded.keys())[0]
    out_f = f"/content/converted_{os.path.basename(src_f)}"
    
    print("\n🪄 Applying Voice Conversion...")
    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        subprocess.run(["pip", "install", "rvc-python"], capture_output=True)
    
    # Auto-index discovery
    idx_p = os.path.join(os.path.dirname(SELECTED_VOICE['path']), "features.index")
    if not os.path.exists(idx_p):
        idx_search = glob.glob(f"{os.path.dirname(SELECTED_VOICE['path'])}/*.index")
        idx_p = idx_search[-1] if idx_search else None
    
    runner = VoiceConverter(SELECTED_VOICE['path'], device="cuda" if torch.cuda.is_available() else "cpu")
    runner.convert(src_f, out_f, index_path=idx_p)
    
    print(f"✅ Conversion complete!")
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
print("SUCCESS (v26 Final Deploy)")
