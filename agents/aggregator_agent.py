# agents/aggregator_agent.py
class AggregatorAgent:
    @staticmethod
    def run(question, sub_answers):
        if not sub_answers:
            return "No answer found."
        return " ".join(sub_answers)