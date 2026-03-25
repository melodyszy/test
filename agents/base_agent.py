from core.llm import llm

class BaseAgent:

    def __init__(self, system_prompt, user_prompt):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

    def run(self, memory, context):

        history = "\n".join([m["content"] for m in memory])

        prompt = self.user_prompt.format(
            history=history,
            query=context.get("query", ""),
            refined=context.get("refined", ""),
            docs=context.get("docs", "")
        )

        messages = memory + [{
            "role": "user",
            "content": prompt
        }]

        return llm.chat(self.system_prompt, messages)