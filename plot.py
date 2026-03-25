# visualize_hotpotqa_metrics_safe.py

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =======================
# 1️⃣ 加载结果文件
# =======================
output_file = Path("hotpotqa_results_deepseek_sdk_cached.json")

if not output_file.exists():
    raise FileNotFoundError(f"结果文件不存在，请检查路径: {output_file}")

with open(output_file, "r", encoding="utf-8") as f:
    results = json.load(f)

# =======================
# 2️⃣ 提取指标，自动处理 AE 是字典或列表的情况
# =======================
def extract_numeric(value):
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, dict) and "score" in value:
        return float(value["score"])
    elif isinstance(value, list) and len(value) > 0:
        # 取第一个元素当作指标
        return float(value[0])
    else:
        # fallback 0
        return 0.0

ae_scores = [extract_numeric(r["AE"]) for r in results]
diversity_scores = [extract_numeric(r["Diversity"]) for r in results]
oversearch_scores = [extract_numeric(r["Over-search"]) for r in results]
stability_scores = [extract_numeric(r["Stability"]) for r in results]

num_questions = len(results)
x = np.arange(1, num_questions + 1)

# =======================
# 3️⃣ 每题指标曲线
# =======================
plt.figure(figsize=(14,6))
plt.plot(x, ae_scores, label="AE", marker='o', markersize=3)
plt.plot(x, diversity_scores, label="Diversity", marker='x', markersize=3)
plt.plot(x, oversearch_scores, label="Over-search", marker='s', markersize=3)
plt.plot(x, stability_scores, label="Stability", marker='^', markersize=3)

plt.xlabel("Question Index")
plt.ylabel("Score")
plt.title("HotpotQA Benchmark Metrics per Question")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =======================
# 4️⃣ 指标分布直方图
# =======================
plt.figure(figsize=(12,8))
plt.hist(ae_scores, bins=20, alpha=0.5, label="AE")
plt.hist(diversity_scores, bins=20, alpha=0.5, label="Diversity")
plt.hist(oversearch_scores, bins=20, alpha=0.5, label="Over-search")
plt.hist(stability_scores, bins=20, alpha=0.5, label="Stability")

plt.xlabel("Score")
plt.ylabel("Number of Questions")
plt.title("HotpotQA Metrics Distribution")
plt.legend()
plt.show()

# =======================
# 5️⃣ 平均值统计
# =======================
print(f"平均 AE: {np.mean(ae_scores):.3f}")
print(f"平均 Diversity: {np.mean(diversity_scores):.3f}")
print(f"平均 Over-search: {np.mean(oversearch_scores):.3f}")
print(f"平均 Stability: {np.mean(stability_scores):.3f}")