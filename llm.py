from openai import OpenAI


class LLM:
    def __init__(self, api_key, model):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model

    # ✅ 简单清理（可选，其实可以不要）
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        return text.strip()

    def generate(self, prompt, max_tokens=256):
        # ✅ 不再使用 ascii 编码
        prompt = self.clean_text(prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )

            text = response.choices[0].message.content

            # ✅ 防止 None 或奇怪返回
            if not text:
                return ""

            return text.strip()

        except Exception as e:
            print("OpenRouter Error:", e)

            # ❗ 调试阶段建议直接抛出
            raise e

            # 如果你想保底而不是中断，用这个：
            # return "fallback query"