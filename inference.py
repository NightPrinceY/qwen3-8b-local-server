import time
from openai import OpenAI


class QwenInference:
    def __init__(self, base_url: str = "http://localhost:8000/v1"):
        self.client = OpenAI(base_url=base_url, api_key="none")
        self.model = "qwen3-8b"

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        return response.choices[0].message.content

    def generate_with_stats(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
        )
        elapsed = time.perf_counter() - start

        tokens = response.usage.completion_tokens
        return {
            "text": response.choices[0].message.content,
            "tokens": tokens,
            "elapsed": elapsed,
            "tok_per_sec": tokens / elapsed,
        }
