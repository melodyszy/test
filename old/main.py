# improved_hotpotqa_deepseek_rag.py
import json
import time
from pathlib import Path
import os
from dotenv import load_dotenv
from openai import OpenAI
import random

# =======================
# 1️⃣ Load DeepSeek API Key
# =======================
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请在 .env 文件中设置 DEEPSEEK_API_KEY")

MODEL_NAME = "deepseek-chat"
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# =======================
# 2️⃣ Load HotpotQA
# =======================
HOTPOTQA_PATH = Path("hotpot_dev.json")
with open(HOTPOTQA_PATH, "r", encoding="utf-8") as f:
    hotpot_data = json.load(f)

AE_test_set = hotpot_data[:100]  # 可以改为全部7405条

# =======================
# 3️⃣ Improved Retriever
# =======================
class ImprovedRetriever:
    """
    改进版本：简单语义匹配 + top-k扩展
    """
    def __init__(self, data, top_k=12):
        self.data = data
        self.top_k = top_k
        self.docs = []
        for item in data:
            for title, content in item["context"]:
                title_str = " ".join(title) if isinstance(title, list) else str(title)
                content_str = " ".join(content) if isinstance(content, list) else str(content)
                self.docs.append(f"{title_str}: {content_str}")

    def search(self, query):
        query_lower = query.lower()
        candidates = []
        for doc in self.docs:
            doc_lower = doc.lower()
            # 简单匹配每个单词出现次数作为 score
            score = sum(1 for w in query_lower.split() if w in doc_lower) / max(len(query_lower.split()),1)
            if score > 0:
                candidates.append((score, doc))
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = [doc for _, doc in candidates[:self.top_k]]
        return results if results else ["No relevant doc found"]

# =======================
# 4️⃣ Improved RAG Agent
# =======================
class ImprovedRAGAgent:
    def __init__(self, client, model, retriever):
        self.client = client
        self.model = model
        self.retriever = retriever

    def step(self, question, history, max_retries=3, max_tokens=350):
        context_docs = self.retriever.search(question)
        formatted_docs = "\n".join([f"Doc {i+1}: {d}" for i, d in enumerate(context_docs)])
        prompt = f"""
You are a careful AI assistant.
Question: {question}
Context documents (use only these):
{formatted_docs}
History of previous reasoning steps: {history}
Instructions:
1. Reason step by step, using ONLY the provided context.
2. If answer is missing, try to infer using the context.
3. Provide intermediate reasoning clearly.
4. End with: Final Answer: <your answer>
"""
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=max_tokens
                )
                content = resp.choices[0].message.content.strip()
                return {"query": question, "reasoning": content, "docs": context_docs}
            except Exception as e:
                print(f"⚠️ DeepSeek request failed (attempt {attempt+1}): {e}")
                time.sleep(2)
        return {"query": question, "reasoning": "API request failed", "docs": context_docs}

# =======================
# 5️⃣ DeepSeek Judge
# =======================
class LLMJudgeDeepSeek:
    def __init__(self, client, model=MODEL_NAME, max_retries=3):
        self.client = client
        self.model = model
        self.max_retries = max_retries

    def judge(self, question, trajectory, judge_last_n=3):
        last_steps = trajectory[-judge_last_n:] if len(trajectory) >= judge_last_n else trajectory
        scores = []
        for step in last_steps:
            reasoning = step["reasoning"]
            docs = step["docs"]
            prompt = f"""
You are an AI judge. Evaluate the answer reasoning.
Question: {question}
Answer reasoning: {reasoning}
Context docs: {docs}
Rules:
- Provide 3 scores (0~1) separated by commas: correctness,relevance,completeness
- Correctness: 1 if fully accurate, 0 if completely wrong
- Relevance: 1 if fully uses context, 0 if unrelated
- Completeness: 1 if fully addresses question, 0 if missing info
Return ONLY 3 numbers.
"""
            for attempt in range(self.max_retries):
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=128
                    )
                    text = resp.choices[0].message.content.strip()
                    c, r, g = [float(x.strip()) for x in text.split(",")]
                    scores.append([c, r, g])
                    break
                except Exception:
                    time.sleep(1)
            else:
                scores.append([0,0,0])
        avg_scores = [sum(x)/len(x) for x in zip(*scores)]
        return avg_scores

# =======================
# 6️⃣ Benchmark Loop
# =======================
retriever = ImprovedRetriever(AE_test_set, top_k=12)  # top_k 提高
agent = ImprovedRAGAgent(client, MODEL_NAME, retriever)
judge = LLMJudgeDeepSeek(client, MODEL_NAME)

all_results = []
max_steps = 7

for idx, item in enumerate(AE_test_set):
    question = item["question"]
    expected = item["answer"]
    trajectory = []
    history = []

    for step_id in range(max_steps):
        step = agent.step(question, history)
        step["step_id"] = step_id
        trajectory.append(step)
        history.append(step)
        if "Final Answer" in step.get("reasoning","") or "Answer:" in step.get("reasoning",""):
            break

    avg_c, avg_r, avg_g = judge.judge(question, trajectory, judge_last_n=3)
    all_results.append({
        "question": question,
        "expected": expected,
        "trajectory": trajectory,
        "Judge": [avg_c, avg_r, avg_g]
    })

    print(f"[{idx+1}/{len(AE_test_set)}] C={avg_c:.2f}, R={avg_r:.2f}, G={avg_g:.2f}")

with open("hotpotqa_results_deepseek_improved.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("✅ Benchmark 完成，结果已保存到 hotpotqa_results_deepseek_improved.json")