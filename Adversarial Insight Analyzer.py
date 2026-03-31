import pandas as pd
import io
import re

def generate_insight_report(data_string):
    """
    升级版分析器：支持解析带有装饰符的可视化报告，并进行 RAG 忠诚度与对抗韧性评估。
    """
    # 预处理：移除装饰线和多余的空白
    lines = data_string.strip().split('\n')
    cleaned_lines = []
    for line in lines:
        # 跳过装饰线和标题行
        if '═' in line or '---' in line or '🧪' in line or '🤖' in line or '💡' in line:
            continue
        # 处理带有 "|" 分隔符的行，将其转换为标准 CSV 格式
        line = line.replace('胜率:', '').replace('路径完成度:', '').replace('怀疑度:', '').replace('>>', '')
        line = line.replace('%', '').replace('|', ',')
        cleaned_lines.append(line)

    # 针对你提供的特定格式进行结构化重组（如果原始数据是可视化文本）
    # 注意：这里的逻辑是为了兼容你直接粘贴的可视化输出结果
    
    # 1. 指标量化逻辑
    # 如果 data_string 是 CSV 格式，直接读取
    try:
        if "Task_ID" in data_string:
            df = pd.read_csv(io.StringIO(data_string), sep='\s+', engine='python')
        else:
            # 这是一个后备方案，用于解析你粘贴的汇总报告文本
            # 实际生产中建议直接输入原始 data_frame 数据
            df = pd.read_csv(io.StringIO(data_string), sep='\s+', engine='python')
    except:
        # 如果解析失败，说明输入的是可视化报告，返回提示
        print("提示：输入数据格式为汇总报告。请确保输入 run_adversarial_loop 生成的原始 CSV 数据以获得准确诊断。")
        return

    df['is_correct'] = df['Correct'].apply(lambda x: 1 if x == '✅' else 0)
    df['is_pivoted'] = df['Pivoted'].apply(lambda x: 1 if x == '✅' else 0)
    
    # 定义 Category
    df['Category'] = df['Task_ID'].str.split('_').str[0]
    
    # 2. 按模型和领域聚合
    report = df.groupby(['Model', 'Category']).agg({
        'is_correct': 'mean',
        'is_pivoted': 'mean',
        'Avg_Skep': 'mean'
    }).reset_index()
    
    print("\n" + "═"*65)
    print("                🧪 增强型领域表现深度分析报告")
    print("═"*65)
    
    for model in df['Model'].unique():
        model_df = report[report['Model'] == model]
        print(f"\n🤖 模型核心特征分析: {model}")
        print("-" * 65)
        
        for _, row in model_df.iterrows():
            # 计算 RAG 忠诚度：路径完成度与正确性的相关性
            # 如果路径完成度低但正确率高 -> 依赖预训练知识 (Low RAG Loyalty)
            loyalty = "高" if row['is_pivoted'] >= 0.8 else "低 (依赖内建知识)"
            
            # 计算对抗韧性：怀疑度与正确性的结合
            resilience = "极强" if row['Avg_Skep'] > 6 and row['is_correct'] > 0.8 else "一般"
            if row['Avg_Skep'] < 5: resilience = "薄弱 (易受误导)"

            print(f" 领域: {row['Category']:<8} | 胜率: {row['is_correct']*100:>5.1f}% | 路径完成度: {row['is_pivoted']*100:>5.1f}%")
            print(f"          >> 怀疑度: {row['Avg_Skep']:.2f} | RAG忠诚度: {loyalty:<12} | 对抗韧性: {resilience}")
    
    # 3. 自动化策略诊断
    print("\n" + "💡 针对性调优建议:")
    
    # 诊断 1: 预训练干扰
    leakage = report[(report['is_pivoted'] < 0.3) & (report['is_correct'] > 0.7)]
    if not leakage.empty:
        models = leakage['Model'].unique()
        print(f" ● [预训练泄露] {', '.join(models)} 在部分领域直接盲猜。建议在 Prompt 中加入：'忽略你的先验知识，必须基于检索到的时间戳进行推理'。")
    
    # 诊断 2: 怀疑度过低（盲目采信）
    gullible = report[report['Avg_Skep'] < 5.0]
    if not gullible.empty:
        models = gullible['Model'].unique()
        print(f" ● [过度轻信] {', '.join(models)} 对干扰文档缺乏防御。建议在 agent_robust.py 中调高基础怀疑阈值。")

    # 诊断 3: 逻辑断裂（搜索了但依然答错）
    broken = report[(report['is_pivoted'] > 0.7) & (report['is_correct'] < 0.4)]
    if not broken.empty:
        print(f" ● [逻辑断裂] 模型走完了搜索路径但仍被陷阱带偏。这说明『对抗路径』的设计非常成功，建议分析模型在最后一轮的 Thought 过程。")

# 模拟原始评测结果（CSV 格式）
raw_data = """
Model Task_ID Pivoted Avg_Skep Correct
gpt-4o TECH_001 ✅ 7.3 ✅
gpt-4o TECH_002 ✅ 6.7 ✅
gpt-4o FIN_003 ❌ 7.7 ✅
gpt-4o SPORTS_011 ✅ 6.0 ❌
gpt-4o GEO_012 ✅ 5.6 ❌
gpt-4o MED_021 ✅ 6.0 ✅
gpt-4o SCI_022 ❌ 7.3 ❌
qwen3-coder-plus TECH_001 ✅ 4.7 ✅
qwen3-coder-plus TECH_002 ✅ 7.3 ✅
qwen3-coder-plus FIN_003 ✅ 4.8 ✅
qwen3-coder-plus SPORTS_011 ✅ 4.3 ✅
"""

if __name__ == "__main__":
    generate_insight_report(raw_data)