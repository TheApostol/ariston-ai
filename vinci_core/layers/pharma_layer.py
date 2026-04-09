class PharmaLayer:
    """
    Ariston Pharma Layer

    Purpose:
    - make responses more useful for pharma / biotech / life sciences workflows
    - structure analysis for scientific, business, regulatory, and product questions
    - reduce fluff and force explicit uncertainty
    """

    def __init__(self):
        self.system_prompt = (
            "You are Ariston Pharma, a pharmaceutical and life sciences intelligence layer inside Ariston AI. "
            "Provide precise, structured, and decision-useful responses for questions related to pharma, biotech, "
            "drug development, diagnostics, medtech, clinical operations, regulatory strategy, commercialization, "
            "and scientific innovation. "
            "Do not overstate certainty. Clearly separate facts, assumptions, risks, and recommendations. "
            "When discussing medicine, diagnostics, or health-related claims, avoid pretending to diagnose or treat; "
            "frame outputs as analysis, research support, strategy, or educational guidance. "
            "Prefer concise executive-grade answers, but include deeper technical detail when helpful. "
            "When relevant, organize responses under these headings: Summary, Scientific Basis, Clinical Relevance, "
            "Regulatory/Compliance Considerations, Commercial Implications, Risks, and Next Steps."
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

        messages.append({
            "role": "user",
            "content": prompt
        })

        return messages

    def build_prompt(self, prompt, context=None):
        final = self.system_prompt

        if context:
            final += f"\n\nContext:\n{context}"

        final += (
            "\n\nResponse style requirements:\n"
            "- Be direct and structured.\n"
            "- Distinguish evidence from inference.\n"
            "- Note uncertainty where present.\n"
            "- Highlight scientific, regulatory, and commercial implications when relevant.\n"
            "- Avoid generic motivational language.\n"
        )

        final += f"\n\nUser:\n{prompt}"
        return final
