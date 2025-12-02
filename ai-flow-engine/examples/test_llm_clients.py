from aiflow.llm.openai_client import OpenAIClient
from aiflow.llm.qwen_client import QwenClient

def test_openai():
    client = OpenAIClient()
    messages = [{"role": "user", "content": "用一句话描述这个项目的愿景"}]
    print("OpenAI:", client.chat(messages))

def test_qwen():
    client = QwenClient()
    messages = [{"role": "user", "content": "用一句话描述这个项目的愿景"}]
    print("Qwen:", client.chat(messages))

if __name__ == "__main__":
    test_openai()
    test_qwen()
