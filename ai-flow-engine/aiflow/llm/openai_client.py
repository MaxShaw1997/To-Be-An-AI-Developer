import os
from openai import OpenAI
from .base import LLMClient

class OpenAIClient(LLMClient):
    def __init__(self, model="gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message["content"]

    def embed(self, texts, **kwargs):
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            **kwargs
        )
        return [item["embedding"] for item in response.data]
