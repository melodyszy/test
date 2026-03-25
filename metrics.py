from llm import LLM


# =========================
# LLM Judge
# =========================
class LLMJudge:
    def __init__(self, api_key, model):
        self.llm = LLM(api_key, model)

    def judge(self, question, reasoning, docs):
        prompt = f"""
You are an evaluator.

Question:
{question}

Step reasoning:
{reasoning}

Docs:
{docs}

Score from 0 to 1:

correctness:
relevance:
groundedness:

Return EXACT format:
correctness: X
relevance: X
groundedness: X
"""

        resp = self.llm.generate(prompt)
        return self.parse(resp)

    def parse(self, text):
        try:
            lines = text.lower().split("\n")

            c = float([l for l in lines if "correctness" in l][0].split(":")[1])
            r = float([l for l in lines if "relevance" in l][0].split(":")[1])
            g = float([l for l in lines if "groundedness" in l][0].split(":")[1])

            return c, r, g

        except:
            return 0.0, 0.0, 0.0


# =========================
# Step Scores
# =========================
def compute_step_scores(question, trajectory, judge):
    scores = []
    useful = []

    for step in trajectory:
        c, r, g = judge.judge(
            question,
            step["reasoning"],
            step["docs"]
        )

        score = 0.5 * c + 0.3 * r + 0.2 * g

        scores.append(score)
        useful.append(1 if score > 0.3 else 0)

    return scores, useful


# =========================
# Adaptation Efficiency
# =========================
def compute_adaptation_efficiency(scores, useful_flags):
    if not scores:
        return {"error_step": -1, "AE": 0, "useful_ratio": 0}

    error_step = 0
    for i, s in enumerate(scores):
        if s < 0.5:
            error_step = i
            break

    recovery = sum(scores[error_step:]) / len(scores)
    useful_ratio = sum(useful_flags) / len(useful_flags)

    return {
        "error_step": error_step,
        "AE": recovery,
        "useful_ratio": useful_ratio
    }


# =========================
# Extra Metrics（论文用🔥）
# =========================
def compute_oversearch(scores):
    first_correct = None
    for i, s in enumerate(scores):
        if s > 0.8:
            first_correct = i
            break

    if first_correct is None:
        return 0

    return (len(scores) - first_correct - 1) / len(scores)


def compute_query_diversity(trajectory):
    queries = [s["query"] for s in trajectory]
    return len(set(queries)) / len(queries)


def compute_stability(scores):
    if len(scores) < 2:
        return 0
    diffs = [abs(scores[i] - scores[i - 1]) for i in range(1, len(scores))]
    return sum(diffs) / len(diffs)