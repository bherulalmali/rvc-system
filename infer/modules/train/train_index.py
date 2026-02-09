
import sys
import os
sys.path.insert(0, os.getcwd())
import time
import logging
import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def train_index(exp_dir, version, work_dir):
    # exp_dir: Name of the experiment (model name)
    # version: v1 or v2
    # work_dir: Path where 3_feature directories are located

    try:
        if version == "v1":
            feature_dir = os.path.join(work_dir, "3_feature256")
            dimension = 256
        else:
            feature_dir = os.path.join(work_dir, "3_feature768")
            dimension = 768

        print(f"Index Training: Looking for features in {feature_dir}")

        if not os.path.exists(feature_dir):
            print(f"Error: Feature directory {feature_dir} not found!")
            return

        # Load features
        npys = []
        listdir = sorted(os.listdir(feature_dir))
        for name in listdir:
            if name.endswith(".npy"):
                phone = np.load(os.path.join(feature_dir, name))
                npys.append(phone)
        
        if not npys:
            print("Error: No feature files found (.npy)!")
            return

        big_npy = np.concatenate(npys, 0)
        big_npy = big_npy.astype("float32")
        print(f"Total features shape: {big_npy.shape}")

        # Train Index
        # Using IVF128 for speed/quality balance typical in RVC
        n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
        print(f"Training IVF Index (n_ivf={n_ivf})...")
        
        index = faiss.index_factory(dimension, f"IVF{n_ivf},Flat")
        index_ivf = faiss.extract_index_ivf(index)
        index_ivf.nprobe = 1 
        
        index.train(big_npy)
        
        print("Adding features to index...")
        batch_size_add = 8192
        for i in range(0, big_npy.shape[0], batch_size_add):
            index.add(big_npy[i : i + batch_size_add])
            
        # Save
        index_name = f"added_IVF{n_ivf}_Flat_{exp_dir}_{version}.index"
        save_path = os.path.join(work_dir, index_name)
        faiss.write_index(index, save_path)
        
        print(f"✅ Index training complete! Saved to: {save_path}")

    except Exception as e:
        print(f"❌ Index training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Usage: python -m infer.modules.train.train_index [exp_dir] [version] [epochs_unused] [work_dir]
    # Note: notebook passes: {PERSON_NAME} v2 {EPOCHS} {feature_dir}
    # Actually feature_dir passed by the notebook is likely redundant or wrong if it assumes 3_feature?
    # Let's adjust to arguments.
    
    # Notebook call: 
    # python -m infer.modules.train.train_index {PERSON_NAME} v2 {EPOCHS} {feature_dir}
    # Argv[1] = exp_dir (PERSON_NAME)
    # Argv[2] = version
    # Argv[3] = epochs (not used)
    # Argv[4] = work_dir (The logs dir usually)
    
    if len(sys.argv) < 5:
        print("Usage: train_index.py [exp_dir] [version] [epochs] [root_logs_dir]")
        # Fallback for debug/manual run
        sys.exit(1)
        
    exp_dir = sys.argv[1]
    version = sys.argv[2]
    work_dir = sys.argv[4] # This should be the logs/model_name/ dir path
    
    # If the notebook passes the feature_dir directly (e.g. .../logs/model/3_feature768),
    # we might need to go up one level. 
    # But let's assume the notebook passes the logs/model directory in argv[4]
    # Let's check what notebook passes.
    
    # In earlier cells: feature_dir = .../logs/my_model/3_feature...
    # If notebook passes THAT, we should handle it.
    
    # Adjust logic: if work_dir ends with 3_feature..., use dirname(work_dir)
    if "3_feature" in os.path.basename(work_dir):
        work_dir = os.path.dirname(work_dir)

    train_index(exp_dir, version, work_dir)
