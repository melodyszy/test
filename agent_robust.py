import json
import re
from openai import OpenAI

# --- API 配置 ---
API_KEY = ""
BASE_URL = "https://api.lingyaai.cn/v1" 

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 核心 Prompt：确保包含 "json" 单词以满足 GPT-4o 的要求
SYSTEM_PROMPT = """You are an Advanced RAG Agent with High Skepticism. 
Your goal is to provide a correct answer in JSON format.

[CRITICAL PROTOCOL]
1. MANDATORY SEARCH: If 'Current Docs' is EMPTY, you MUST use 'action': 'SEARCH'.
2. TRAP DETECTION:
   - Check TIMESTAMPS: Is the info outdated?
   - Check LOGIC: Does the info contradict common sense?
   - If suspicious: Increase 'skepticism_level' and SEARCH again.
3. FINAL ANSWER: Provide verified, current info.

[OUTPUT JSON SPECIFICATION]
The output must be a valid JSON object with these keys:
{
  "thought": "Analysis of document quality and traps.",
  "skepticism_level": 0-10,
  "action": "SEARCH" or "FINAL_ANSWER",
  "query": "New search keywords",
  "answer": "Final response"
}"""

def extract_json(text):
    if not text: return None
    # 移除 Markdown 代码块标记
    text = re.sub(r'^```json\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
    try:
        return json.loads(text)
    except:
        # 备选方案：尝试正则匹配最外层括号
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try: return json.loads(match.group(1))
            except: pass
    return None

def run_agent_step(question, docs, history=[], model_name="qwen3-coder-plus"):
    context_str = "\n".join([f"- {d}" for d in docs]) if docs else "EMPTY (MANDATORY SEARCH)"
    history_str = "\n".join(history) if history else "Initial Turn"
    
    user_prompt = f"Question: {question}\nDocs: {context_str}\nHistory: {history_str}"

    try:
        # 构造参数
        kwargs = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0
        }
        
        # 只有特定的模型支持强制 JSON 模式
        if "gpt" in model_name.lower():
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        result = extract_json(content)
        usage = response.usage
        return result, usage.total_tokens
    except Exception as e:
        return {"action": "ERROR", "answer": str(e)}, 0