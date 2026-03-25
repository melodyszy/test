# conversational_ae_valid.py

import json
import re
import os
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

# =======================
# 3️⃣ Retriever
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

# 先初始化（用于过滤）
retriever = Retriever(data)

# =======================
# 4️⃣ valid filtering
# =======================
def is_valid_structure(item):
    facts = item.get("supporting_facts", [])
    if len(facts) < 2:
        return False
    titles = [f[0] for f in facts]
    if len(set(titles)) < 2:
        return False
    return True

def is_valid_context(item):
    facts = item.get("supporting_facts", [])
    context_titles = set([t for t,_ in item.get("context", [])])
    for title, _ in facts:
        if title not in context_titles:
            return False
    return True

def is_retrievable(item, retriever):
    facts = item.get("supporting_facts", [])
    titles = [f[0] for f in facts[:2]]
    query = " ".join(titles)
    docs = retriever.search(query)

    hit = 0
    for t in titles:
        if any(t.lower() in d.lower() for d in docs):
            hit += 1

    return hit >= 1

def is_valid_item(item, retriever):
    if not is_valid_structure(item):
        return False
    if not is_valid_context(item):
        return False
    if not is_retrievable(item, retriever):
        return False
    return True

# =======================
# 5️⃣ 过滤数据
# =======================
filtered_dataset = []

for item in data:
    if is_valid_item(item, retriever):
        filtered_dataset.append(item)

print(f"原始样本: {len(data)}")
print(f"过滤后样本: {len(filtered_dataset)}")

dataset = filtered_dataset[:50]

# =======================
# 6️⃣ 多轮对话生成（🔥核心）
# =======================
def build_conversation(item):

    question = item["question"]
    facts = item.get("supporting_facts", [])

    title1 = facts[0][0]
    title2 = facts[1][0]

    return [
        f"What is {title1}?",
        f"What is the relation between {title1} and {title2}?",
        f"Using previous information, answer: {question}"
    ]

# =======================
# 7️⃣ 生成答案
# =======================
def generate_answer(question, history, docs):

    hist_text = "\n".join(history)
    docs_text = "\n".join(docs)

    prompt = f"""
You are a QA assistant.

Conversation history:
{hist_text}

Current question:
{question}

Documents:
{docs_text}

Answer ONLY based on documents.

Output:
Final Answer: <answer>
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
# 8️⃣ Judge（修复版）
# =======================
def judge(question, answer):

    prompt = f"""
Return ONLY 0 or 1.

1 = correct
0 = wrong

Question: {question}
Answer: {answer}
"""

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )

    text = resp.choices[0].message.content.strip()

    return 1 if text.startswith("1") else 0

# =======================
# 9️⃣ AE计算
# =======================
def compute_conversation_ae(success_turn):

    if success_turn == -1:
        return 0

    return 1 / (success_turn + 1)

# =======================
# 🔟 主循环
# =======================
results = []

for idx, item in enumerate(dataset):

    conversation = build_conversation(item)

    history = []
    success_turn = -1

    print(f"\n===== Dialog {idx+1} =====")

    for turn_id, q in enumerate(conversation):

        docs = retriever.search(q)
        answer = generate_answer(q, history, docs)

        correct = judge(item["question"], answer)

        print(f"[Turn{turn_id}] Q: {q}")
        print(f"[Turn{turn_id}] A: {answer}")
        print(f"[Turn{turn_id}] Correct={correct}")

        history.append(f"Q: {q}\nA: {answer}")

        if correct == 1 and success_turn == -1:
            success_turn = turn_id

    ae = compute_conversation_ae(success_turn)

    print(f"👉 AE={ae:.3f}")

    results.append({
        "conversation": conversation,
        "AE": ae
    })

# =======================
# 保存
# =======================
with open("conv_ae_valid.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ 完成")