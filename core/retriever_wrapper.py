# core/retriever_wrapper.py
from core.query import MsgTextQuery

class RetrieverWrapper:
    def __init__(self, dataset=None):
        # 可以传 dataset，也可为空
        self.query_tool = MsgTextQuery(dataset)

    def search(self, question, top_k=5):
        expr, keywords = self.query_tool.question(question)
        # 简单返回 expr.query_str 模拟检索结果
        return [expr.query_str]