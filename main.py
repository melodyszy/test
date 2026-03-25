# main.py
import json
from core.memory_manager import Memory
from core.retriever_wrapper import RetrieverWrapper
from agents import answer_agent, judge_agent
from metrics.ae import compute_ae

# -----------------------
# 加载数据
# -----------------------
dataset_file = "data/hotpot_dev.json"
with open(dataset_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)[:50]

# -----------------------
# 初始化 Retriever
# -----------------------
retriever = RetrieverWrapper(dataset)

# -----------------------
# 主循环
# -----------------------
results = []

for idx, item in enumerate(dataset):
    question = item.get("question","")  # ✅ 必须在循环里定义
    print(f"\n===== Question {idx+1}: {question} =====")
    
    # 初始化 memory
    memory = Memory(uid=f"user{idx}", memory_id=f"mem{idx}")
    memory_text = memory.get()
    
    # 自动检索 docs
    docs = retriever.search(question)  # ✅ 此时 question 已经定义
    
    # 调用 AnswerAgent
    answer = answer_agent.AnswerAgent.run({
        "question": question,
        "memory": memory_text,
        "docs": docs
    })
    
    # 保存到 memory
    memory.insert("assistant", answer)
    
    # JudgeAgent 打分
    scores = judge_agent.JudgeAgent.run(answer, item.get("answer",""), docs)
    
    # AE 计算
    ae = compute_ae([scores])
    
    print(f"Answer: {answer}")
    print(f"Scores per step: {scores}")
    print(f"👉 AE={ae:.3f}")
    
    results.append({
        "question": question,
        "answer": answer,
        "scores": scores,
        "AE": ae
    })

# -----------------------
# 保存结果
# -----------------------
with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ Benchmark completed.")