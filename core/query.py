# core/query.py
import re
import logging
import json

# 简化 MatchTextExpr / MatchDenseExpr
class MatchTextExpr:
    def __init__(self, fields, query_str, top_n=100, meta=None):
        self.fields = fields
        self.query_str = query_str
        self.top_n = top_n
        self.meta = meta or {}

class MatchDenseExpr:
    def __init__(self, name, vector, dtype="float", metric="cosine", topk=10, meta=None):
        self.name = name
        self.vector = vector
        self.dtype = dtype
        self.metric = metric
        self.topk = topk
        self.meta = meta or {}

# 简化 term_weight 和 synonym
class TermWeightDealer:
    def weights(self, tokens, preprocess=False):
        return [(t, 1.0) for t in tokens]
    def split(self, txt):
        return txt.split()

class SynonymDealer:
    def lookup(self, tk):
        return []  # 简化不返回同义词

class MsgTextQuery:

    def __init__(self, dataset=None):
        self.dataset = dataset  # 可以接收 dataset
        self.tw = TermWeightDealer()
        self.syn = SynonymDealer()
        self.query_fields = ["content"]

    @staticmethod
    def add_space_between_eng_zh(text: str):
        return text

    @staticmethod
    def rmWWW(txt: str):
        return txt.replace("www.", "").replace("http://", "").replace("https://", "")

    def is_chinese(self, txt: str):
        return any('\u4e00' <= c <= '\u9fff' for c in txt)

    @staticmethod
    def sub_special_char(txt: str):
        return re.sub(r"[^\w\s]", "", txt)

    def question(self, txt, tbl="messages", min_match: float=0.6):
        original_query = txt
        txt = self.add_space_between_eng_zh(txt)
        txt = re.sub(r"[ :|\r\n\t,，。？?/`!！&^%%()\[\]{}<>]+", " ", txt).strip()
        txt = self.rmWWW(txt)

        # 简化处理，直接返回 MatchTextExpr
        return MatchTextExpr(
            self.query_fields,
            query_str=txt,
            top_n=100,
            meta={"minimum_should_match": min_match, "original_query": original_query}
        ), txt.split()[:32]