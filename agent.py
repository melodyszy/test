from llm import LLM


# ✅ 简单“伪真实”检索器（替代 Dummy）
class SimpleRetriever:
    def __init__(self):
        self.knowledge = [
            "The current President of the United States is Joe Biden.",
            "Joe Biden is a member of the Democratic Party.",
            "The US has two major parties: Democratic and Republican.",
        ]

    def search(self, query):
        results = []

        for doc in self.knowledge:
            if any(word.lower() in doc.lower() for word in query.split()):
                results.append(doc)

        # 至少返回一点内容，避免空
        return results if results else ["No relevant information found."]


def clean_query(text):
    if not text:
        return ""
    text = text.replace('"', '').replace("**", "").strip()
    return text.split("\n")[0][:80]


def is_answered(text):
    if not text:
        return False

    text = text.lower()

    return (
        "final answer" in text
        or ("(" in text and ")" in text and len(text) < 100)
    )


class RAGAgent:
    def __init__(self, api_key, model):
        self.llm = LLM(api_key, model)
        self.retriever = SimpleRetriever()  # ✅ 替换掉 Dummy

    def step(self, question, history):

        # ===== Early Stop =====
        last_reasoning = ""
        if isinstance(history, list) and len(history) > 0:
            if isinstance(history[-1], dict):
                last_reasoning = history[-1].get("reasoning", "")

        if is_answered(last_reasoning):
            return {
                "query": "",
                "docs": [],
                "reasoning": last_reasoning
            }

        # ===== Query Planning =====
        query_prompt = f"""
You are a search planner.

Original question:
{question}

Previous steps:
{history}

Generate ONE search query.

Requirements:
- Stay close to the original question
- Must be different from previous queries
- Max 10 words
- No quotation marks

Only output the query.
"""

        raw_query = self.llm.generate(query_prompt)
        query = clean_query(raw_query)

        if not query:
            query = question[:50]

        # ===== Retrieval =====
        docs = self.retriever.search(query)

        # ===== Reasoning（关键修复）=====
        reasoning_prompt = f"""
Answer the question.

Question: {question}
Docs: {docs}

Rules:
- Use docs if helpful
- If docs are insufficient, use your knowledge
- Do NOT ask for more documents
- Be concise

If confident:
    Final Answer: <answer>

Otherwise:
    Next Step: <what to search>

No extra text.
"""

        reasoning = self.llm.generate(reasoning_prompt).strip()

        return {
            "query": query,
            "docs": docs,
            "reasoning": reasoning
        }