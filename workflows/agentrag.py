from core.memory import Memory
from core.recovery import RecoveryRunner
from agents import refiner_agent, answer_agent, planner_agent, aggregator_agent, judge_agent
from config import MAX_TURN

class AgentRAG:
    def __init__(self, retriever):
        self.retriever = retriever
        self.runner = RecoveryRunner(max_retries=3)

    def run(self, question, gt):
        memory = Memory()
        subtasks = planner_agent.run(question)

        all_scores = []
        sub_answers = []

        for sub in subtasks:

            context_query = sub

            for step in range(MAX_TURN):

                # 🔥 1. Refiner（带恢复）
                ref_res = self.runner.run(
                    refiner_agent.run,
                    memory.get(),
                    context_query
                )
                if not ref_res.success:
                    break

                refined = ref_res.data

                # 🔥 2. Retrieval（带恢复）
                ret_res = self.runner.run(
                    self.retriever.search,
                    refined
                )
                if not ret_res.success:
                    break

                docs = ret_res.data

                # 🔥 3. Answer（带恢复）
                ans_res = self.runner.run(
                    answer_agent.run,
                    memory.get(),
                    refined,
                    docs
                )
                if not ans_res.success:
                    break

                answer = ans_res.data

                # memory
                memory.add("user", refined)
                memory.add("assistant", answer)

                # 🔥 4. Judge（带恢复）
                judge_res = self.runner.run(
                    judge_agent.run,
                    answer,
                    gt,
                    docs
                )

                score = judge_res.data if judge_res.success else [0,0,0]
                all_scores.append(score)

                print(f"[Subtask:{sub} Step{step}] {score}")

                # ✅ 成功恢复
                if score[0] >= 0.8:
                    sub_answers.append(answer)
                    break

                # 🔥 主动恢复：修改 query（关键！！！）
                context_query = self._recover_query(refined, answer, docs)

        # 🔥 final aggregation
        final_answer = aggregator_agent.run(question, sub_answers)
        final_score = judge_agent.run(final_answer, gt, "")
        all_scores.append(final_score)

        return all_scores

    # 🔥 核心恢复策略
    def _recover_query(self, query, answer, docs):
        return f"""
{query}

Previous answer seems incorrect.
Try another way:
- expand entity
- use biography / nationality / relation keywords
- re-search with different wording
"""