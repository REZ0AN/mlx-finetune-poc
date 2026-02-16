import os
from mlx_lm import load, stream_generate

MODEL_PATH = "mlx-community/Qwen2.5-3B-Instruct-4bit"
ADAPTER_PATH = "adapters"

print("=" * 60)
print("Interactive Streaming - Fine-tuned Qwen2.5")
print("=" * 60)

if not os.path.exists(ADAPTER_PATH):
    print(f"\n[ERROR] No adapters at {ADAPTER_PATH}")
    exit(1)

print("\n[LOAD] Loading model...")
model, tokenizer = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
print("[OK] Model loaded!\n")

while True:
    user_input = input("\n💬 You: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    # Qwen2.5 ChatML format
    prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
    
    print("\n🤖 Assistant: ", end="", flush=True)

    # stream_generate yields a response object for every token
    for response in stream_generate(model, tokenizer, prompt, max_tokens=2000):
        # response is an object with a 'text' attribute
        print(response.text, end="", flush=True)
            
    print("\n" + ("-" * 60))