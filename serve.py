"""
Local multi-GPU LLM server using vLLM.
Exposes an OpenAI-compatible API at http://localhost:8000

RTX 2080 Ti notes (CC 7.5):
- No Flash Attention 2 (needs CC 8.0+)
- Uses FlashInfer with JIT compilation via ninja
- awq_marlin needs CC 8.0+ (Ampere) — use plain awq instead
- enforce-eager required (torch.compile Inductor unstable on CC 7.5)
- fp16 only (bf16 is software emulated, too slow)
- tensor_parallel must divide num_attention_heads evenly
"""
import subprocess
import sys
import os

MODEL = "Qwen/Qwen3-8B-AWQ"

# GPUs 1,3,4,6 — cleanest (skip GPU0=occupied, GPU2=4GB used, GPU5=partial)
# tensor_parallel=4: Qwen3-8B has 32 heads → 32/4=8 per GPU ✓
GPUS = "1,3,4,6"
TENSOR_PARALLEL = 4

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = GPUS
env["PATH"] = f"{os.environ['HOME']}/.local/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
env["HF_TOKEN"] = env.get("HF_TOKEN", "")
env["NCCL_DEBUG"] = "WARN"

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL,
    "--tensor-parallel-size", str(TENSOR_PARALLEL),
    "--dtype", "float16",
    "--quantization", "awq",
    "--max-model-len", "8192",
    "--gpu-memory-utilization", "0.85",
    "--enforce-eager",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--served-model-name", "qwen3-8b",
]

print(f"Starting vLLM server: {MODEL}")
print(f"GPUs: {GPUS} | Tensor parallel: {TENSOR_PARALLEL}")
print(f"Endpoint: http://localhost:8000/v1")
print("-" * 50)

subprocess.run(cmd, env=env)
