# agents/judge_agent.py
from collections import Counter

class JudgeAgent:
    @staticmethod
    def run(answer, ground_truth, docs=None):
        """
        简单的 token-overlap 判分器
        返回 [C,R,G]，0~1
        """
        def tokenize(text):
            return [t.lower() for t in text.replace("*","").replace(",","").split() if t]

        ans_tokens = tokenize(answer)
        gt_tokens = tokenize(ground_truth)

        if not ans_tokens or not gt_tokens:
            return [0.0,0.0,0.0]

        # Content correctness C: token overlap / GT tokens
        C = len(set(ans_tokens) & set(gt_tokens)) / len(set(gt_tokens))

        # Recall R: token overlap / Answer tokens
        R = len(set(ans_tokens) & set(gt_tokens)) / len(set(ans_tokens))

        # Grammar G: 简化，非空答案就算1
        G = 1.0 if answer.strip() else 0.0

        return [round(C,2), round(R,2), round(G,2)]