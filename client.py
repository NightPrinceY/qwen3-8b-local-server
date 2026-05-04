"""
Benchmark client for the local vLLM server.
Run this after serve.py is up.
"""
from inference import QwenInference

PROMPTS = [
    "Explain what RAG (Retrieval Augmented Generation) is in 3 sentences.",
    "Write a Python function that reverses a linked list.",
    "What are the key differences between LoRA and full fine-tuning?",
]

if __name__ == "__main__":
    llm = QwenInference()

    print("=" * 60)
    print("vLLM Local Server — Benchmark")
    print("=" * 60)

    total_tokens, total_time = 0, 0.0
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}] {prompt[:60]}...")
        result = llm.generate_with_stats(prompt)
        print(f"Response: {result['text'][:200]}...")
        print(f"Tokens: {result['tokens']} | Time: {result['elapsed']:.2f}s | Speed: {result['tok_per_sec']:.1f} tok/s")
        total_tokens += result["tokens"]
        total_time += result["elapsed"]

    print("\n" + "=" * 60)
    print(f"Average speed: {total_tokens / total_time:.1f} tok/s")
    print("=" * 60)
