from core.llm import llm

class RefinerAgent:
    @staticmethod
    def run(inputs):
        question = inputs.get("question","")
        memory = inputs.get("memory","")
        prompt = f"Refine the question using history:\n{memory}\n{question}"
        return llm.chat("RefinerAgent", [{"role":"user","content":prompt}])