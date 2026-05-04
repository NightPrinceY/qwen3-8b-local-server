# Qwen3-8B-AWQ Local Server

OpenAI-compatible LLM server running **Qwen3-8B-AWQ** locally via **vLLM** across 4x RTX 2080 Ti GPUs.

## Hardware

| Spec | Value |
|---|---|
| GPUs | 4x NVIDIA RTX 2080 Ti (11 GB VRAM each) |
| Total VRAM | ~44 GB across tensor-parallel group |
| CUDA | 12.9 |
| Quantization | AWQ (INT4) |
| Precision | FP16 |

> RTX 2080 Ti is CC 7.5 (Turing) — no Flash Attention 2, no `awq_marlin`. Uses FlashInfer + `--enforce-eager`.

## Setup

```bash
git clone https://github.com/NightPrinceY/qwen3-8b-local-server
cd qwen3-8b-local-server

cp .env.example .env
# Add your HuggingFace token to .env

pip install vllm openai-whisper
```

## Usage

**Start the server:**
```bash
python serve.py
```

Server starts at `http://localhost:8000/v1` (OpenAI-compatible).

**Run benchmark:**
```bash
python client.py
```

**Run Islamic knowledge test:**
```bash
python islamic_test.py
```

## Inference API

`inference.py` provides a reusable `QwenInference` class for querying the server from any script.

```python
from inference import QwenInference

llm = QwenInference()

# Simple generation
text = llm.generate("Explain transformers in 2 sentences.")
print(text)

# With a system prompt
text = llm.generate(
    "What is Zakat?",
    system="You are a knowledgeable Islamic scholar.",
    max_tokens=300,
    temperature=0.3,
)

# With speed stats
result = llm.generate_with_stats("Write a Python quicksort.")
print(result["text"])
print(f"{result['tok_per_sec']:.1f} tok/s")
```

### `generate()`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | required | User message |
| `system` | `str \| None` | `None` | System prompt |
| `max_tokens` | `int` | `512` | Max output tokens |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `thinking` | `bool` | `False` | Enable Qwen3 chain-of-thought |

Returns `str`.

### `generate_with_stats()`

Same parameters as `generate()`. Returns a `dict`:

```python
{
    "text": str,        # model output
    "tokens": int,      # completion tokens used
    "elapsed": float,   # wall-clock seconds
    "tok_per_sec": float
}
```

## Benchmark Results

Tested on 4x RTX 2080 Ti with `--enforce-eager`:

| Test | Prompts | Tokens | Speed |
|---|---|---|---|
| General benchmark | 3 | 1,375 | **10.7 tok/s** |
| Islamic knowledge | 6 | 3,600 | **10.9 tok/s** |

## Configuration

Key settings in `serve.py`:

| Parameter | Value | Reason |
|---|---|---|
| `--tensor-parallel-size` | 4 | Qwen3-8B has 32 heads → 8 per GPU |
| `--quantization` | awq | awq_marlin needs CC 8.0+ |
| `--dtype` | float16 | bf16 is emulated on CC 7.5 |
| `--enforce-eager` | true | torch.compile unstable on CC 7.5 |
| `--gpu-memory-utilization` | 0.85 | leaves headroom for KV cache |
| `CUDA_VISIBLE_DEVICES` | 1,3,4,6 | skips occupied GPUs |

## Qwen3 Thinking Mode

Qwen3 supports chain-of-thought reasoning. It is **disabled by default** in `inference.py`. To enable it:

```python
result = llm.generate("Solve this step by step: ...", thinking=True)
```

When enabled, the model outputs a `<think>...</think>` block before the final answer.

## Environment Variables

```bash
HF_TOKEN=hf_your_token_here          # Required to download the model
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  # Adjust to your free GPUs
```

## Author

**Yahya Alnwsany** — [GitHub](https://github.com/NightPrinceY) · [HuggingFace](https://huggingface.co/NightPrince)
