import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from config import MODEL_PROVIDER, DEEPSEEK_MODEL, OLLAMA_MODEL

load_dotenv()

class LLM:
    def __init__(self):
        self.provider = MODEL_PROVIDER.lower()
        if self.provider=="deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key: raise ValueError("请在 .env 文件里设置 DEEPSEEK_API_KEY")
            self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        elif self.provider=="ollama":
            self.client = None
        else:
            raise ValueError(f"未知模型提供商: {self.provider}")

    def chat(self, system_prompt, messages, temperature=0.0):
        if self.provider=="deepseek":
            full_messages = [{"role":"system","content":system_prompt}] if system_prompt else []
            full_messages.extend(messages)
            resp = self.client.chat.completions.create(model=DEEPSEEK_MODEL, messages=full_messages, temperature=temperature)
            return resp.choices[0].message.content.strip()
        elif self.provider=="ollama":
            prompt_text = (system_prompt+"\n" if system_prompt else "") + (messages[-1]["content"] if messages else "")
            url = "http://localhost:11434/api/generate"
            payload = {"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False}
            resp = requests.post(url,json=payload)
            return resp.json().get("response","")

llm = LLM()