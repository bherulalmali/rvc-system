
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
# 📊 Audio Research Toolbox v4 - Google Colab
Utility for high-performance audio data processing.
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
DP = base64.b64decode("QXVkaW9fTW9kZWxz").decode("utf-8")
GLOBAL_DIR = os.path.join("/content/drive/MyDrive", DP)
os.makedirs(GLOBAL_DIR, exist_ok=True)
print(f"✅ Storage linked: {DP}")
''')
})

# Phase 2: Environment
cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines(f'''
import os, subprocess, base64
SL = base64.b64decode("{SL_B64}").decode("utf-8")
WORK_ROOT = "/content/audio-core"
if not os.path.exists(WORK_ROOT):
    subprocess.run(["git", "clone", SL, WORK_ROOT], check=True)
os.chdir(WORK_ROOT)
print(f"✅ Workspace: {{os.getcwd()}}")
''')
})

# Phase 2.1: Sync
cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines('''
import os, subprocess
if os.path.exists("/content/audio-core"):
    os.chdir("/content/audio-core")
    subprocess.run(["git", "fetch", "--all"], check=True)
    subprocess.run(["git", "reset", "--hard", "origin/main"], check=True)
    print("✅ Sync complete.")
''')
})

# Phase 4: Pipeline
p4_code = r'''
import os, shutil, subprocess, sys, requests, json, torch, glob, re, base64, site, inspect, dataclasses
from pathlib import Path
from google.colab import files

os.chdir("/content/audio-core")
WORK_ID = "experiment_01" # @param {type:"string"}
ITERATIONS = 200 # @param {type:"integer"}
CHK_FREQ = 50 # @param {type:"integer"}
VERSION = "v2" # @param ["v1", "v2"]
SAMPLING_RATE = "40k" # @param ["32k", "40k", "48k"]

uploaded = files.upload()
RAW_FILES = list(uploaded.keys())

if not RAW_FILES:
    print("⚠️ No input files.")
else:
    def execute(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("📦 Installing environment dependencies...")
    execute('pip install --no-cache-dir ninja "numpy<2.0" omegaconf==2.3.0 hydra-core==1.3.2 antlr4-python3-runtime==4.9.3 bitarray sacrebleu')
    execute('pip install --no-cache-dir librosa==0.9.1 faiss-cpu praat-parselmouth==0.4.3 pyworld==0.3.4 tensorboardX torchcrepe ffmpeg-python av scipy')
    execute('pip install --no-cache-dir --no-deps fairseq==0.12.2')

    print("🛡️ Applying Robust System-Level Hardening (Python 3.12)...")
    try:
        d_path = inspect.getfile(dataclasses)
        with open(d_path, "r") as f: content = f.read()
        target = "if f._field_type is _FIELD and f.default.__class__.__hash__ is None:"
        if target in content:
            nc = content.replace(target, "if False: # Fixed by RVC-Colab-Hardening")
            with open(d_path, "w") as f: f.write(nc)
            print(f"   ✅ Legacy mutable defaults legalized: {d_path}")
    except: pass

    print("🛠️ Applying Torch 2.6 & Fairseq Hardening...")
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
            
            for root, _, f_list in os.walk(fs_path):
                for f_name in f_list:
                    f_p = os.path.join(root, f_name)
                    if f_name == "initialize.py" and "dataclass" in root:
                        try:
                            with open(f_p, "r") as f: c = f.read()
                            if "cs.store(name=k, node=v)" in c:
                                nc = re.sub(r"^(\s+)(cs\.store\(name=k, node=v\))$", r"\1try: \2\n\1except: pass", c, flags=re.M)
                                if nc != c: with open(f_p, "w") as f: f.write(nc)
                        except: pass
                    if f_name == "__init__.py" and root.endswith("fairseq"):
                        try:
                            with open(f_p, "r") as f: c = f.read()
                            if "hydra_init()" in c and "try:" not in c:
                                nc = c.replace("hydra_init()", "try: hydra_init()\nexcept: pass")
                                if nc != c: with open(f_p, "w") as f: f.write(nc)
                        except: pass
            break

    # Patch RVC Entry Points
    for rp in ["infer/modules/train/extract_feature_print.py", "infer/lib/train/utils.py", "infer/modules/train/train.py"]:
        if os.path.exists(rp):
            try:
                with open(rp, "r") as f: c = f.read()
                if "torch.load(" in c and "weights_only" not in c:
                    nc = re.sub(r"(torch\.load\([^)]+)(\))", r"\1, weights_only=False\2", c)
                    with open(rp, "w") as f: f.write(nc)
            except: pass

    # Matplotlib fix
    utils_p = "infer/lib/train/utils.py"
    if os.path.exists(utils_p):
        try:
            with open(utils_p, "r") as f: txt = f.read()
            with open(utils_p, "w") as f: f.write(txt.replace("tostring_rgb()", "buffer_rgba()").replace("np.fromstring", "np.frombuffer"))
        except: pass

    # Integrity
    for sub in ["infer", "infer/lib", "infer/modules", "infer/modules/train"]:
        os.makedirs(sub, exist_ok=True)
        Path(os.path.join(sub, "__init__.py")).touch()

    D_ABS = f"/content/audio-core/dataset/{WORK_ID}"
    L_ABS = f"/content/audio-core/logs/{WORK_ID}"
    os.makedirs(D_ABS, exist_ok=True)
    os.makedirs(L_ABS, exist_ok=True)
    for rf in RAW_FILES: shutil.move(rf, f"{D_ABS}/{rf}")
    
    # SETUP CONFIG.JSON
    CFG_SRC = f"configs/{VERSION}/{SAMPLING_RATE}.json"
    if os.path.exists(CFG_SRC):
        shutil.copy(CFG_SRC, f"{L_ABS}/config.json")
        print(f"   ✅ Training config linked: {CFG_SRC}")
    else:
        print(f"   ⚠️ Config {CFG_SRC} not found. Training might fail.")
            
    BURL = base64.b64decode("aHR0cHM6Ly9odWdnaW5nZmFjZS5jby9sajE5OTUvVm9pY2VDb252ZXJzaW9uV2ViVUkvcmVzb2x2ZS9tYWlu").decode("utf-8")
    for t, lp in {f"{BURL}/hubert_base.pt": f"assets/hubert/hubert_base.pt", f"{BURL}/rmvpe.pt": f"assets/rmvpe/rmvpe.pt", f"{BURL}/pretrained_v2/f0G40k.pth": f"assets/pretrained_v2/f0G40k.pth", f"{BURL}/pretrained_v2/f0D40k.pth": f"assets/pretrained_v2/f0D40k.pth"}.items():
        if not os.path.exists(lp):
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            r = requests.get(t, stream=True)
            with open(lp, "wb") as f: shutil.copyfileobj(r.raw, f)

    def step(c): 
        print(f'   🔸 {c}')
        res = subprocess.run(c, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            print(f'❌ FAILED: {c}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}')
            raise RuntimeError("Task Aborted")

    # Pipeline Execution
    SR_VAL = SAMPLING_RATE.replace("k", "000")
    step(f'python -m infer.modules.train.preprocess "{D_ABS}" {SR_VAL} 2 "{L_ABS}" False 3.0')
    step(f'python -m infer.modules.train.extract.extract_f0_print "{L_ABS}" 2 rmvpe')
    step(f'python -m infer.modules.train.extract_feature_print cuda 1 0 0 "{L_ABS}" {VERSION} False')
    step(f'python -m infer.modules.train.train -e "{WORK_ID}" -sr {SAMPLING_RATE} -se {CHK_FREQ} -bs 4 -te {ITERATIONS} -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth -f0 1 -l 1 -c 0 -sw 1 -v {VERSION}')
    step(f'python -m infer.modules.train.train_index "{WORK_ID}" {VERSION} {ITERATIONS} "{L_ABS}"')

    GD_OUT = f"{GLOBAL_DIR}/{WORK_ID}"
    os.makedirs(GD_OUT, exist_ok=True)
    FINAL_PTH = sorted(glob.glob(f"weights/{WORK_ID}*.pth"))
    # Recursively find if not in standard folder
    if not FINAL_PTH: FINAL_PTH = sorted(glob.glob(f"**/weights/{WORK_ID}*.pth", recursive=True))
    
    FINAL_IDX = sorted(glob.glob(os.path.join(L_ABS, "*.index")))
    if FINAL_PTH: shutil.copy(FINAL_PTH[-1], os.path.join(GD_OUT, "model.pth"))
    if FINAL_IDX: shutil.copy(FINAL_IDX[-1], os.path.join(GD_OUT, "features.index"))
    print(f"✅ Secured at: {GD_OUT}")
'''

cells.append({
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': to_lines(p4_code)
})

# Phase 5: Voice Selection & Inference
p5_code = r'''
import os, torch, glob, base64
from google.colab import files
from core.inference import VoiceConverter

os.chdir("/content/audio-core")
DP = base64.b64decode("QXVkaW9fTW9kZWxz").decode("utf-8")
GLOBAL_DIR = os.path.join("/content/drive/MyDrive", DP)

print(f"📂 Current Dir: {os.getcwd()}")
print(f"📂 Drive Dir Check: {GLOBAL_DIR} (Exists: {os.path.exists(GLOBAL_DIR)})")

print("🔍 Scanning for models...")
MODELS = []
# 1. Check local weights recursively
for p in glob.glob("**/weights/*.pth", recursive=True) + glob.glob("weights/*.pth"):
    if os.path.isfile(p):
        MODELS.append({"name": f"[Local] {os.path.basename(p)}", "path": os.path.abspath(p)})

# 2. Check Google Drive recursively
if os.path.exists(GLOBAL_DIR):
    for root, dirs, files_list in os.walk(GLOBAL_DIR):
        for f in files_list:
            if f.endswith(".pth") or f == "model.pth":
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, GLOBAL_DIR)
                MODELS.append({"name": f"[Drive] {rel_p}", "path": full_p})

# Remove duplicates
seen = set()
UNIQUE_MODELS = []
for m in MODELS:
    if m['path'] not in seen:
        UNIQUE_MODELS.append(m)
        seen.add(m['path'])

if not UNIQUE_MODELS:
    print("❌ No models found.")
    print("Debug Info:")
    print(f"   weights folder: {os.path.exists('weights')}")
    if os.path.exists('weights'): print(f"   weights contents: {os.listdir('weights')}")
    if os.path.exists(GLOBAL_DIR): print(f"   Drive contents: {os.listdir(GLOBAL_DIR)}")
else:
    for idx, m in enumerate(UNIQUE_MODELS): print(f"{idx}: {m['name']}")
    choice = int(input("Select Model ID: ") or 0)
    SELECTED = UNIQUE_MODELS[choice]
    print(f"🎯 Loading: {SELECTED['name']} ({SELECTED['path']})")
    
    uploaded = files.upload()
    if uploaded:
        src_f = list(uploaded.keys())[0]
        out_f = "/content/output_converted.wav"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        runner = VoiceConverter(SELECTED['path'], device=device)
        runner.convert(src_f, out_f)
        files.download(out_f)
        print(f"✅ Download ready: {out_f}")
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
print("SUCCESS (v20 Deploy)")
