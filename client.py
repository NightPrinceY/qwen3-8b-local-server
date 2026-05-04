"""
Test client for the local vLLM server.
Run this after serve.py is up.
"""
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

def chat(prompt: str, max_tokens: int = 512) -> tuple[str, float]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    elapsed = time.perf_counter() - start
    text = response.choices[0].message.content
    tokens = response.usage.completion_tokens
    return text, tokens, elapsed

if __name__ == "__main__":
    prompts = [
        "Explain what RAG (Retrieval Augmented Generation) is in 3 sentences.",
        "Write a Python function that reverses a linked list.",
        "What are the key differences between LoRA and full fine-tuning?",
    ]

    print("=" * 60)
    print("vLLM Local Server — Benchmark")
    print("=" * 60)

    total_tokens, total_time = 0, 0.0
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}] {prompt[:60]}...")
        text, tokens, elapsed = chat(prompt)
        tps = tokens / elapsed
        print(f"Response: {text[:200]}...")
        print(f"Tokens: {tokens} | Time: {elapsed:.2f}s | Speed: {tps:.1f} tok/s")
        total_tokens += tokens
        total_time += elapsed

    print("\n" + "=" * 60)
    print(f"Average speed: {total_tokens/total_time:.1f} tok/s")
    print("=" * 60)
