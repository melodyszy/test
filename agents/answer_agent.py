# agents/answer_agent.py
from core.llm import llm
from core.memory_manager import Memory
from core.retriever_wrapper import RetrieverWrapper

class AnswerAgent:
    @staticmethod
    def run(inputs):
        question = inputs.get("question","")
        memory_text = inputs.get("memory","")
        docs = inputs.get("docs", [])
        # 构造 prompt
        prompt = f"""
Use the following documents and chat history to answer the question:
Memory:
{memory_text}

Question:
{question}

Documents:
{docs}

Answer:
"""
        # 调用 LLM
        answer = llm.chat("AnswerAgent", [{"role":"user","content":prompt}])
        return answer