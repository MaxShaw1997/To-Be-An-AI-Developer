from abc import ABC, abstractmethod

class LLMClient(ABC):
    """所有模型的抽象基类，统一 chat / embed 接口"""

    @abstractmethod
    def chat(self, messages, **kwargs):
        pass

    @abstractmethod
    def embed(self, texts, **kwargs):
        pass
