# ae_valid_step_final.py

import json
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# =======================
# 1️⃣ API
# =======================
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
MODEL = "deepseek-chat"

# =======================
# 2️⃣ Load Data
# =======================
with open("hotpot_dev.json", "r", encoding="utf-8") as f:
    data = json.load(f)

AE_test_set = data[:100]

# =======================
# 3️⃣ Retriever（TF-IDF）
# =======================
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Retriever:
    def __init__(self, data, top_k=8):
        self.top_k = top_k
        self.docs = []

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

# =======================
# 4️⃣ Answer Generator
# =======================
def generate_answer(question, docs):

    docs_text = "\n".join([f"Doc{i+1}: {d}" for i,d in enumerate(docs)])

    prompt = f"""
Answer using ONLY the documents.

Question: {question}

Documents:
{docs_text}

Think step by step.

Output:
Final Answer: <answer>
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    raw = resp.choices[0].message.content

    match = re.search(r"Final Answer:\s*(.*)", raw)
    return match.group(1) if match else raw

# =======================
# 5️⃣ Query Refinement
# =======================
def refine_query(question, trajectory):

    history = ""
    for t in trajectory:
        history += f"Query: {t['query']}\nAnswer: {t['answer']}\n"

    prompt = f"""
Previous attempts failed.

Improve the search query.

Question: {question}

History:
{history}

Return ONLY a better query.
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    return resp.choices[0].message.content.strip()

# =======================
# 6️⃣ Judge（稳定版）
# =======================
def judge(question, answer, docs):

    prompt = f"""
Evaluate answer.

Return ONLY 3 numbers between 0 and 1.

Format:
0.8,0.9,1.0

Question: {question}
Answer: {answer}
Docs: {docs}
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

    print("⚠️ Judge解析失败:", text)
    return [0,0,0]

# =======================
# 7️⃣ 有效步判断（核心）
# =======================
def is_valid_step(curr_scores, prev_scores):

    c, r, g = curr_scores

    # ✔️ 检索相关
    if r >= 0.5:
        return True

    # ✔️ 有提升
    if prev_scores:
        pc, pr, pg = prev_scores
        if c > pc or g > pg:
            return True

    return False

# =======================
# 8️⃣ AE计算（🔥最终版）
# =======================
def compute_ae(success_valid_step, step_id):

    if success_valid_step <= 0:
        return 0

    # ⭐ Step0直接成功 → 满分
    if step_id == 0:
        return 1.0

    # ⭐ 恢复成功 → 打折
    return (1 / success_valid_step) * 0.8

# =======================
# 9️⃣ 主循环
# =======================
retriever = Retriever(AE_test_set)

all_results = []

for idx, item in enumerate(AE_test_set):

    question = item["question"]

    trajectory = []
    query = question

    valid_steps = 0
    success_valid_step = -1
    prev_scores = None

    max_steps = 5

    print(f"\n====== Q{idx+1} ======")

    for step in range(max_steps):

        docs = retriever.search(query)
        answer = generate_answer(question, docs)

        scores = judge(question, answer, docs)
        c, r, g = scores

        # 🔥 有效步判断
        if is_valid_step(scores, prev_scores):
            valid_steps += 1
            valid_flag = "✅"
        else:
            valid_flag = "❌"

        print(f"[Step{step}] {valid_flag} C={c:.2f}, R={r:.2f}, G={g:.2f}")

        trajectory.append({
            "step": step,
            "query": query,
            "answer": answer,
            "scores": scores,
            "valid": valid_flag
        })

        # 🔥 恢复成功
        if c >= 0.8:
            success_valid_step = valid_steps
            break

        prev_scores = scores

        # 🔥 失败 → 改 query
        query = refine_query(question, trajectory)

    # 🔥 AE计算（已修复）
    ae = compute_ae(success_valid_step, step)

    print(f"👉 AE={ae:.3f}")

    all_results.append({
        "question": question,
        "trajectory": trajectory,
        "AE": ae
    })

# =======================
# 保存
# =======================
with open("ae_final_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\n✅ AE Benchmark 完成")