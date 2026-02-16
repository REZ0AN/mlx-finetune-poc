import json
from tqdm import tqdm
import sys
import os

# Take subset for PoC (adjust this number as needed)
NUM_EXAMPLES = 100


def load_data(file_path='alpaca_data.json'):
    """Load the Alpaca dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data



def format_data(data, num_examples=NUM_EXAMPLES):
    """Format the data for MLX fine-tuning."""
    subset = data[:num_examples]  # Take a subset for quick testing
    formatted_data = []
    for item in tqdm(subset, desc="      Processing"):
        instruction = item['instruction']
        input_text = item.get('input', '')
        output = item['output']
    
        if input_text:
            text = f"<|user|>\n{instruction}\n{input_text}<|end|>\n<|assistant|>\n{output}<|end|>"
        else:
            text = f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{output}<|end|>"
        
        formatted_data.append({"text": text})
    return formatted_data

def save_jsonl(data, file_path='train_data.jsonl'):
    """Save the formatted data as a JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def main():
    data_file = 'alpaca_data.json'
    sample_size = NUM_EXAMPLES
    print("[LOAD] Loading dataset...")
    data = load_data(data_file)
    
    print(f"[FORMAT] Formatting {sample_size} examples...")
    formatted_data = format_data(data, num_examples=sample_size)
    
    print("[SAVE] Saving formatted data to 'train_data.jsonl'...")
    save_jsonl(formatted_data)
    
    print("[OK] Data preparation complete! 'train_data.jsonl' is ready for fine-tuning.")

if __name__ == "__main__":
    main()