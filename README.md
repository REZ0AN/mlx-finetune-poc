# MLX Fine-tuning PoC

A proof-of-concept project for fine-tuning the Qwen2.5-3B-Instruct model using Meta's MLX framework on macOS with 8GB RAM constraints.

## 🎯 Project Overview

This project demonstrates how to:
- Prepare and format training data for LLM fine-tuning
- Use LoRA (Low-Rank Adaptation) techniques with a pre-quantized 4-bit model
- Fine-tune a language model efficiently on resource-constrained hardware
- Perform inference with the adapted model

**Target Model**: `Qwen2.5-3B-Instruct-4bit` (pre-quantized for 8GB RAM)


## Prerequisites
- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11
- 8GB+ RAM

## Setup

```bash
# Clone the repo and navigate to it
cd mlx-finetune-poc

# Run the setup script
bash setup.sh

# Activate the environment
source mlx-poc/bin/activate
```

## Workflow

### 1. Data Preparation

**Script**: `prepare_dataset.py`

Converts the Alpaca dataset into JSONL format with custom prompt templates:

```bash
python prepare_dataset.py
```

**Output**: `train_data.jsonl` (100 examples by default)

**Features**:
- Loads Alpaca JSON dataset
- Formats with instruction/input/output structure
- Supports conditional input handling
- Saves to JSONL format with progress bar

### 2. Model Fine-tuning

**Script**: `finetune_model.py`

Fine-tunes the pre-quantized model using LoRA adapters:

```bash
python finetune_model.py
```

**Configuration**:
```python
TRAIN_CONFIG = {
    "model": "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "data": "./",
    "iters": 100,              # Training iterations
    "batch_size": 1,           # Batch size (constrained by RAM)
    "max_seq_length": 512,     # Maximum sequence length
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "save_every": 50,
    "adapter_path": "./adapters",
}
```

**Key RAM Optimizations**:
- `--grad-checkpoint`: Gradient checkpointing to save memory
- Pre-quantized 4-bit model
- Batch size of 1
- Max sequence length of 512 tokens

### 3. Interactive Testing

**Script**: `testing_model.py`

Chat interface with the fine-tuned model:

```bash
python testing_model.py
```

**Features**:
- Streaming token generation
- Qwen2.5 ChatML format prompts
- Type `quit` or `exit` to exit
- Real-time token output

## 🔧 Technologies Used

| Component | Purpose |
|-----------|---------|
| **MLX** | Apple Silicon optimized ML framework |
| **Transformers** | Model loading and tokenization |
| **LoRA** | Parameter-efficient fine-tuning |
| **Python 3.11** | Runtime |
| **Hugging Face Hub** | Model distribution |

## 📈 Areas for Engineering Improvement

### 1. **Data Pipeline**
- **Issue**: `prepare_dataset.py` uses hardcoded 100 examples
- **Improvement**: 
  - Add command-line arguments for sample size
  - Implement data splitting logic (train/val/test ratio)
  - Add data validation and schema checking
  - Support multiple dataset formats beyond Alpaca

### 2. **Configuration Management**
- **Issue**: Config values hardcoded in `finetune_model.py`
- **Improvement**:
  - Use YAML/JSON config files for reproducibility
  - Support command-line argument overrides
  - Implement config validation schema
  - Add environment variable support

### 3. **Error Handling & Logging**
- **Issue**: Limited error context and no structured logging
- **Improvement**:
  - Implement comprehensive try-catch blocks with meaningful messages
  - Add structured logging (JSON, multiple levels)
  - Create custom exception classes
  - Log training metrics, loss curves, and checkpoints

### 4. **Model Checkpoint Management**
- **Issue**: No rollback or best-model tracking
- **Improvement**:
  - Track validation loss and save best model
  - Implement checkpoint resumption logic
  - Add automatic cleanup of old checkpoints
  - Save training metadata (hyperparameters, timestamps)

### 5. **Performance Monitoring**
- **Issue**: No visibility into memory usage or training progress
- **Improvement**:
  - Add memory profiling (peak RAM, GPU memory)
  - Track training time and throughput (tokens/sec)
  - Implement early stopping based on validation metrics
  - Export metrics to monitoring dashboard (Weights & Biases, TensorBoard)

## 6. **Testing & Validation**
- **Issue**: `testing_model.py` is manual-only, no automated tests
- **Improvement**:
  - Create unit tests for data pipeline
  - Add integration tests for model loading/inference
  - Implement benchmark suite for performance regression
  - Add output quality metrics (BLEU, ROUGE scores)

## 7. **Documentation**
- **Issue**: Sparse inline comments and no API documentation
- **Improvement**:
  - Add docstrings for all functions
  - Create Jupyter notebook tutorial
  - Document hyperparameter tuning guide
  - Add troubleshooting section for common issues

## 8. **Reproducibility**
- **Issue**: Random seed not fixed, no version pinning
- **Improvement**:
  - Pin exact package versions in `requirements.txt`
  - Set random seeds for deterministic results
  - Document exact hardware/OS used for reference runs
  - Create reproducibility checklist

## 9. **Inference Architecture**
- **Issue**: Single interactive session only
- **Improvement**:
  - Add batch inference mode
  - Implement REST API server (FastAPI)
  - Add request/response logging
  - Support multiple concurrent users

## 10. **Resource Management**
- **Issue**: No automatic cleanup or resource pooling
- **Improvement**:
  - Add memory cleanup between epochs
  - Implement model caching strategy
  - Add graceful shutdown handlers
  - Monitor and alert on resource limits

## Example Usage

## Train the model
```bash
# Activate environment
source mlx-poc/bin/activate

# Prepare data
python prepare_dataset.py

# Fine-tune
python finetune_model.py

# Expected output
# ====== STARTING QLORA TRAINING ======
# Training progress...
# [OK] Training complete
# [TEST] Loading Fine-Tuned Model...
# Response: [Model output]
```

## Test with custom prompt
```bash
python testing_model.py

# > What is machine learning?
# > [Streaming response from model]
```

## Dependencies

See `requirements.txt` for full list. Key packages:
- `mlx >= 0.30.6`
- `transformers >= 4.40.0`
- `huggingface_hub >= 1.4.0`
- `python-dotenv >= 1.0.0`

## Environment Variables

Create a `.env` file if needed for API keys or custom paths:

```bash
# Example .env
HF_TOKEN=your_huggingface_token
ADAPTER_PATH=./adapters
```

## Known Limitations

1. **RAM Constraint**: Limited to 8GB devices; batch size locked at 1
2. **Sequence Length**: Capped at 512 tokens due to inference limitations
3. **Model Size**: 3B parameter model; larger models may not fit
4. **Quantization**: Uses fixed 4-bit quantization; no flexible options
5. **Single-GPU**: No multi-device training support

## References

- [Medium Blog](https://medium.com/@dummahajan/train-your-own-llm-on-macbook-a-15-minute-guide-with-mlx-6c6ed9ad036a)

