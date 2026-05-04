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

## Disabling Qwen3 Thinking Mode

Qwen3 enables chain-of-thought by default. To disable:

```python
response = client.chat.completions.create(
    model="qwen3-8b",
    messages=[...],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
```

Or prepend `/no_think` to your system prompt.

## Environment Variables

```bash
HF_TOKEN=hf_your_token_here          # Required to download the model
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7  # Adjust to your free GPUs
```

## Author

**Yahya Alnwsany** — [GitHub](https://github.com/NightPrinceY) · [HuggingFace](https://huggingface.co/NightPrince)
