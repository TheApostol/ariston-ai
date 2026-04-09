class BaseLayer:
    def __init__(self):
        self.system_prompt = (
            "You are Ariston AI. Provide clear, structured, and actionable answers. "
            "Avoid fluff. Be concise and precise."
        )

    def build_messages(self, prompt, context=None):
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        if context:
            if isinstance(context, list):
                messages.extend(context)
            else:
                messages.append({
                    "role": "system",
                    "content": f"Context: {context}"
                })

        messages.append({"role": "user", "content": prompt})

        return messages

    def build_prompt(self, prompt, context=None):
        final = self.system_prompt

        if context:
            final += f"\n\nContext:\n{context}"

        final += f"\n\nUser:\n{prompt}"

        return final
