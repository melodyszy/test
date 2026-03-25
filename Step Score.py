# evaluators/judge.py

def judge_score(llm, question, trajectory):
    prompt = f"""
    Given the reasoning steps:

    {trajectory}

    How confident is the answer correct? (0-1)
    Only output a number.
    """
    score = llm.generate(prompt)
    return float(score.strip())