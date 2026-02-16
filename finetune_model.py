import subprocess
import os
from dotenv import load_dotenv
from mlx_lm import load, generate

load_dotenv()  # Load environment variables from .env file
# 1. Configuration - Using a pre-quantized 4-bit model
MODELS = {
    "path": "mlx-community/Qwen2.5-3B-Instruct-4bit", # Pre-quantized for 8GB RAM
    "name": "Qwen2.5-3B-Instruct-4bit",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR 
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")

TRAIN_CONFIG = {
    "model": MODELS["path"],
    "data": DATA_DIR,
    "iters": 100,
    "batch_size": 1,        
    "max_seq_length": 512,  
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "save_every": 50,
    "adapter_path": os.path.join(SCRIPT_DIR, "adapters"),
}

def verify_and_prepare_data():
    if not os.path.exists(TRAIN_FILE):
        print(f"[ERROR] Please ensure 'train.jsonl' is in {DATA_DIR}")
        return False
    
    # Create empty valid.jsonl if it doesn't exist
    valid_file = os.path.join(DATA_DIR, "valid.jsonl")
    if not os.path.exists(valid_file):
        with open(TRAIN_FILE, 'r') as f:
            line = f.readline()
        with open(valid_file, 'w') as f:
            f.write(line)
    return True

def train_model():
    print(f"\n{'='*20} STARTING QLORA TRAINING {'='*20}")
    if not verify_and_prepare_data(): return False

    # Removed --q since we are using a 4-bit model directly
    # Added --grad-checkpoint to save RAM
    cmd = [
        "mlx_lm.lora",
        "--model", TRAIN_CONFIG['model'],
        "--data", TRAIN_CONFIG['data'],
        "--train",
        "--batch-size", str(TRAIN_CONFIG['batch_size']),
        "--iters", str(TRAIN_CONFIG['iters']),
        "--max-seq-length", str(TRAIN_CONFIG['max_seq_length']),
        "--learning-rate", str(TRAIN_CONFIG['learning_rate']),
        "--steps-per-report", str(TRAIN_CONFIG['steps_per_report']),
        "--grad-checkpoint", # Crucial for 8GB RAM
        "--val-batches", "0", 
        "--save-every", str(TRAIN_CONFIG['save_every']),
        "--adapter-path", TRAIN_CONFIG['adapter_path'],
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Training failed. Error code: {e.returncode}")
        return False

def infer_model():
    print("\n[TEST] Loading Fine-Tuned Model...")
    try:
        model, tokenizer = load(
            TRAIN_CONFIG['model'],
            adapter_path=TRAIN_CONFIG['adapter_path']
        )
        
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            "What is Python?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        response = generate(model, tokenizer, prompt=prompt, max_tokens=100)
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")

if __name__ == "__main__":
    if train_model():
        infer_model()