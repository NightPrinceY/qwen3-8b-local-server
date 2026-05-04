"""
Islamic queries test against the local vLLM server.
"""
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

QUERIES = [
    {
        "category": "Fiqh",
        "question": "What are the conditions (shurut) that make Salah valid according to the four major madhabs?",
    },
    {
        "category": "Aqeedah",
        "question": "Explain the concept of Tawhid and its three categories: Tawhid al-Rububiyyah, Tawhid al-Uluhiyyah, and Tawhid al-Asma wa al-Sifat.",
    },
    {
        "category": "Quran Sciences",
        "question": "What is the difference between Makki and Madani surahs? Give examples and mention the linguistic/thematic characteristics of each.",
    },
    {
        "category": "Hadith",
        "question": "Explain the hadith classification system: Sahih, Hasan, Da'if, and Mawdu'. What criteria differentiate them?",
    },
    {
        "category": "Islamic Finance",
        "question": "What makes a financial transaction Riba? Explain the difference between Riba al-Fadl and Riba al-Nasi'ah with examples.",
    },
    {
        "category": "Arabic",
        "question": "ما هي أركان الإسلام الخمسة؟ اشرح كل ركن باختصار.",
    },
]

SYSTEM_PROMPT = (
    "You are a knowledgeable Islamic scholar assistant. "
    "Answer questions accurately based on Quran, Sunnah, and classical Islamic scholarship. "
    "Cite sources where possible. Be concise but thorough."
)

def ask(question: str, max_tokens: int = 600) -> tuple[str, int, float]:
    start = time.perf_counter()
    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    elapsed = time.perf_counter() - start
    text = response.choices[0].message.content
    tokens = response.usage.completion_tokens
    return text, tokens, elapsed


if __name__ == "__main__":
    print("=" * 70)
    print("Islamic Knowledge Test — Qwen3-8B-AWQ via vLLM")
    print("=" * 70)

    total_tokens, total_time = 0, 0.0

    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(QUERIES)}] Category: {q['category']}")
        print(f"Q: {q['question']}")
        print(f"{'─' * 70}")

        text, tokens, elapsed = ask(q["question"])
        tps = tokens / elapsed

        print(f"\n{text}")
        print(f"\n[{tokens} tokens | {elapsed:.1f}s | {tps:.1f} tok/s]")

        total_tokens += tokens
        total_time += elapsed

    print(f"\n{'=' * 70}")
    print(f"Total: {total_tokens} tokens | {total_time:.1f}s | Avg {total_tokens/total_time:.1f} tok/s")
    print("=" * 70)
