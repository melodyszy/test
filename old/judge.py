from core.llm import LLM
import re

llm = LLM("你的deepseek key")

def score_state(question, trajectory):
    prompt = f"""
    Question: {question}

    Reasoning steps:
    {trajectory}

    Evaluate how likely the answer is correct.

    Output ONLY a number between 0 and 1.
    Example: 0.2 or 0.85
    """

    output = llm.generate(prompt)

    # 提取数字
    match = re.search(r"0\.\d+|1\.0", output)

    if match:
        return float(match.group())
    else:
        return 0.0