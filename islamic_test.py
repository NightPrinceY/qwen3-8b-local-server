"""
Islamic queries test against the local vLLM server.
"""
from inference import QwenInference

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

if __name__ == "__main__":
    llm = QwenInference()

    print("=" * 70)
    print("Islamic Knowledge Test — Qwen3-8B-AWQ via vLLM")
    print("=" * 70)

    total_tokens, total_time = 0, 0.0
    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(QUERIES)}] Category: {q['category']}")
        print(f"Q: {q['question']}")
        print(f"{'─' * 70}")

        result = llm.generate_with_stats(q["question"], system=SYSTEM_PROMPT, max_tokens=600, temperature=0.3)
        print(f"\n{result['text']}")
        print(f"\n[{result['tokens']} tokens | {result['elapsed']:.1f}s | {result['tok_per_sec']:.1f} tok/s]")

        total_tokens += result["tokens"]
        total_time += result["elapsed"]

    print(f"\n{'=' * 70}")
    print(f"Total: {total_tokens} tokens | {total_time:.1f}s | Avg {total_tokens / total_time:.1f} tok/s")
    print("=" * 70)
