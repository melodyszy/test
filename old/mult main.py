# agentrag_final.py

import json
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

# =======================
# API
# =======================
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"

# =======================
# Load Data
# =======================
with open("hotpot_dev.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = data[:30]

# =======================
# Retriever（TF-IDF）
# =======================
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Retriever:
    def __init__(self, data, top_k=8):
        self.docs = []
        self.top_k = top_k

        for item in data:
            for title, content in item["context"]:
                title = " ".join(title) if isinstance(title, list) else str(title)
                content = " ".join(content) if isinstance(content, list) else str(content)
                self.docs.append(f"{title}: {content}")

        self.vec = TfidfVectorizer(stop_words="english")
        self.doc_vec = self.vec.fit_transform(self.docs)

    def search(self, query):
        q_vec = self.vec.transform([query])
        scores = (self.doc_vec @ q_vec.T).toarray().flatten()
        idx = np.argsort(scores)[-self.top_k:][::-1]
        return [self.docs[i] for i in idx]

retriever = Retriever(dataset)

# =======================
# Subtask Planner
# =======================
def plan_subtasks(question):

    prompt = f"""
Decompose into 2-3 reasoning steps.

Question: {question}

Format:
Step1: ...
Step2: ...
Step3: ...
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    steps = re.findall(r"Step\d+:\s*(.*)", resp.choices[0].message.content)

    return steps if steps else [question]

# =======================
# Query（带history）
# =======================
def build_query(q, history):
    hist = " ".join(history[-2:])
    return q + " " + hist

# =======================
# Answer Generator
# =======================
def generate_answer(q, history, docs):

    prompt = f"""
You are a QA agent.

History:
{history}

Question:
{q}

Docs:
{docs}

Answer using docs.

Final Answer:
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    text = resp.choices[0].message.content

    match = re.search(r"Final Answer:\s*(.*)", text)
    return match.group(1) if match else text

# =======================
# Judge（连续评分）
# =======================
def judge(answer, gt):

    prompt = f"""
Evaluate answer vs ground truth.

Return 3 scores (0~1):
C (correctness)
R (relevance)
G (groundedness)

Format: 0.8,0.9,1.0

Ground Truth: {gt}
Answer: {answer}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    text = resp.choices[0].message.content
    nums = re.findall(r"\d+\.?\d*", text)

    if len(nums) >= 3:
        return list(map(float, nums[:3]))

    return [0,0,0]

def is_correct(scores):
    return scores[0] >= 0.7

# =======================
# AE（连续版）
# =======================
def compute_ae(scores_list):

    total = 0
    weight = 0

    for i, (c,r,g) in enumerate(scores_list):

        s = (c+r+g)/3
        w = 1/(i+1)

        total += s * w
        weight += w

    return total/weight if weight else 0

# =======================
# 主循环（自适应 + subtask）
# =======================
results = []

for idx, item in enumerate(dataset):

    print(f"\n===== Q{idx+1} =====")

    question = item["question"]
    gt = item["answer"]

    subtasks = plan_subtasks(question)

    history = []
    scores_list = []
    success_turn = -1

    max_turns = 6

    for t in range(max_turns):

        if t < len(subtasks):
            q = subtasks[t]
        else:
            q = question

        query = build_query(q, history)
        docs = retriever.search(query)

        ans = generate_answer(q, history, docs)

        scores = judge(ans, gt)
        scores_list.append(scores)

        print(f"[Turn{t}] Q: {q}")
        print(f"[Turn{t}] A: {ans}")
        print(f"[Turn{t}] Scores={scores}")

        history.append(ans)

        if is_correct(scores):
            success_turn = t
            break

    ae = compute_ae(scores_list)

    print(f"👉 AE={ae:.3f}")

    results.append({
        "question": question,
        "AE": ae,
        "scores": scores_list
    })

# 保存
with open("final_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ 完成")