# vinci_core/context/builder.py

def build_context(prompt: str, context: dict | None, layer: str):
    base_context = {
        "prompt": prompt,
        "layer": layer,
    }

    if context:
        base_context.update(context)

    # Layer-specific enrichment (future: FHIR, pharma data, etc.)
    return base_context
