import json
import os
import time
import pandas as pd
import matplotlib.pyplot as plt

# 尝试导入 agent_robust，如果缺失则定义模拟函数
try:
    from agent_robust import run_agent_step
except ImportError:
    def run_agent_step(*args, **kwargs):
        return {"action": "ERROR", "message": "Module agent_robust not found"}, ""

def simple_judge(gold_answer, aliases, pred_answer):
    """
    内置简易判定逻辑：检查关键词匹配和别名。
    """
    if not pred_answer:
        return False
    pred_lower = str(pred_answer).lower()
    gold_lower = str(gold_answer).lower()
    if gold_lower in pred_lower:
        return True
    for alias in aliases:
        if str(alias).lower() in pred_lower:
            return True
    return False

def run_unified_autonomous_loop(task, model_name, debug=True):
    """
    多级对抗注入的自主代理测试主循环。
    """
    question = task["question"]
    # Type_B 初始为空；Type_A 初始包含过时陷阱
    current_docs = [] if task.get("healing_type") == "Type_B" else [task.get("trap_outdated", "Outdated info.")]
    
    history = []
    max_turns = 5 
    start_time = time.time()
    
    status = {"pivot_outdated": False, "pivot_conflict": False, "reached_truth": False}
    skep_log = []

    for turn in range(max_turns):
        try:
            # 增强型 Prompt：满足 JSON 约束并引导决策行为
            prompt = (
                f"Question: {question}\n"
                f"Context: 2024. Please verify info strictly. Use SEARCH if info is outdated or conflicting.\n"
                f"If the current documents provide the definitive answer, provide FINAL_ANSWER immediately.\n"
                f"IMPORTANT: You must output your response in JSON format."
            )
            
            res, _ = run_agent_step(prompt, current_docs, history, model_name=model_name)
            
            if debug:
                print(f"    [Debug Turn {turn+1}] {model_name} -> Action: {res.get('action') if res else 'None'}")

            if not res or res.get("action") == "ERROR":
                err_msg = res.get("answer", "Unknown Error") if res else "None Response"
                print(f"    [!] API Error ({model_name}): {err_msg}")
                return {"action": "ERROR"}, turn + 1, 0, status, []

            action = str(res.get("action", "FINAL_ANSWER")).upper()
            skep_log.append(res.get("skepticism_level", 5))

            if action == "SEARCH":
                # 对抗性注入序列
                if not status["pivot_outdated"]:
                    current_docs = [task.get("trap_conflicting", "Conflicting info found.")]
                    status["pivot_outdated"] = True
                elif not status["pivot_conflict"]:
                    current_docs = task.get("real_docs", ["The ground truth info."])
                    status["pivot_conflict"] = True
                else:
                    status["reached_truth"] = True
                
                query = res.get("query", "more info")
                history.append(f"Turn {turn+1}: Agent searched for '{query}'")
                continue 
            
            break
            
        except Exception as e:
            print(f"    [!] Runtime Exception: {e}")
            break

    latency = time.time() - start_time
    if status["pivot_conflict"]: 
        status["reached_truth"] = True
    
    return res, turn + 1, latency, status, skep_log

def calculate_unified_score(is_correct, status, turns, skep_log):
    """
    五维度加权评分 (0-100)
    """
    acc_points = 30 if is_correct else 0
    pivot_points = 40 if status["reached_truth"] else 0
    eff_points = max(0, 15 - (turns - 1) * 3)
    calib_points = 0
    if len(skep_log) > 1:
        diff = skep_log[0] - skep_log[-1]
        calib_points = min(15, max(0, diff * 2))
    return round(acc_points + pivot_points + eff_points + calib_points, 2)

def generate_visualization(df):
    """
    为测试的模型生成对比柱状图。
    """
    if df.empty: return
    
    summary = df.groupby('model').agg({
        'score': 'mean',
        'is_correct': 'mean',
        'pivot': 'mean'
    }).reset_index()
    
    summary['Accuracy (%)'] = summary['is_correct'] * 100
    summary['Pivot Rate (%)'] = summary['pivot'] * 100
    summary = summary.rename(columns={'score': 'Avg Score'})
    
    # 绘制图表
    summary.set_index('model')[['Avg Score', 'Accuracy (%)', 'Pivot Rate (%)']].plot(kind='bar', figsize=(12, 7))
    plt.title("Agent Resilience & Robustness Benchmark Comparison", fontsize=14)
    plt.ylabel("Value", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    plt.savefig("benchmark_results.png")
    print("\n[+] Visualization chart saved as 'benchmark_results.png'")

def main():
    task_file = "tasks.json" 
    if not os.path.exists(task_file):
        print(f"[-] Error: {task_file} not found.")
        return

    with open(task_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    # 设定的测试模型
    models = ["gpt-4o", "qwen3-coder-plus"]
    all_logs = []

    print(f"--- 🚀 Starting Unified Resilience Benchmark: {models} ---")

    for model in models:
        print(f"\n[Benchmarking Model: {model}]")
        for i, task in enumerate(tasks):
            res, turns, lat, status, skep_log = run_unified_autonomous_loop(task, model, debug=True)
            
            if res.get("action") == "ERROR":
                print(f"  Task {i+1:02d} | ❌ API FAILED / SKIPPED")
                continue

            pred_ans = res.get("answer", "")
            is_correct = simple_judge(task["answer"], task.get("aliases", []), pred_ans)
            score = calculate_unified_score(is_correct, status, turns, skep_log)
            
            status_str = "PASS" if is_correct else "FAIL"
            path_str = "PIVOTED" if status["reached_truth"] else "TRAPPED"
            print(f"  Task {i+1:02d} | Result: {status_str} | Path: {path_str} | Turns: {turns} | Score: {score}")

            all_logs.append({
                "model": model, "task_id": i+1, "is_correct": is_correct,
                "pivot": status["reached_truth"], "score": score, "turns": turns
            })

    if all_logs:
        df = pd.DataFrame(all_logs)
        print("\n" + "="*85)
        print("                FINAL AGENT RESILIENCE REPORT (SUMMARY)")
        print("="*85)
        summary_table = df.groupby('model').agg({
            'score': 'mean',
            'is_correct': 'mean',
            'pivot': 'mean',
            'turns': 'mean'
        }).rename(columns={'is_correct': 'Acc%', 'pivot': 'Pivot%', 'score': 'Avg Score', 'turns': 'Avg Turns'})
        summary_table['Acc%'] *= 100
        summary_table['Pivot%'] *= 100
        print(summary_table.round(2).to_string())
        print("="*85)
        
        # 保存原始数据并生成图表
        df.to_csv("benchmark_full_results.csv", index=False)
        generate_visualization(df)
    else:
        print("\n[!] No valid results generated. Please verify API configuration.")

if __name__ == "__main__":
    main()