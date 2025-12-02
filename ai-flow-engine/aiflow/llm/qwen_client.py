import os
from http import HTTPStatus
from dashscope import Generation, TextEmbedding
from .base import LLMClient

class QwenClient(LLMClient):
    def __init__(self, model="qwen-max"):
        self.api_key = os.getenv("QWEN_API_KEY")
        self.model = model

    def chat(self, messages, **kwargs):
        prompt = "\n".join([m.get("content", "") for m in messages])

        resp = Generation.call(
            model=self.model,
            prompt=prompt,
            api_key=self.api_key,
            **kwargs
        )
        if resp.status_code != HTTPStatus.OK:
            raise RuntimeError(f"Qwen API Error: {resp.message}")

        return resp.output["text"]

    def embed(self, texts, **kwargs):
        resp = TextEmbedding.call(
            model="text-embedding-v2",
            input=texts,
            api_key=self.api_key,
            **kwargs
        )
        return [item["embedding"] for item in resp.output["embeddings"]]
