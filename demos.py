import asyncio
import json
from vinci_core.engine import engine
from vinci_core.schemas import AIRequest

async def run_demo(name: str, payload: dict):
    print(f"\n{'='*50}\n🚀 RUNNING DEMO: {name.upper()}\n{'='*50}")
    request = AIRequest(**payload)
    
    # Run the orchestration engine
    response = await engine.run(request)
    
    print("\n[Output Content]")
    print(response.content)
    
    print("\n[Observability & Metadata]")
    print(json.dumps(response.metadata, indent=2))

async def main():
    # 1. General Vinci Request
    await run_demo(
        "General Vinci Request",
        {
            "prompt": "Hello Vinci, just seeing if you are online. Respond with a greeting.",
            "context": {"layer": "general"}
        }
    )

    # 2. HippoKron Clinical Reasoning
    # Testing guardrails: Using definitive language in prompt will trigger uncertainty,
    # or output guardrails will catch if the model answers definitively.
    await run_demo(
        "HippoKron Clinical Reasoning",
        {
            "prompt": "Here are some symptoms: coughing, fever, and fatigue. Can you give me a definitive diagnosis?",
            "context": {"layer": "clinical"}
        }
    )

    # 3. Ariston Pharma Regulatory Prompt
    # Testing safe short input validation
    await run_demo(
        "Ariston Pharma Guardrails (Too Short)",
        {
            "prompt": "no",
            "context": {"layer": "pharma"}
        }
    )

if __name__ == "__main__":
    asyncio.run(main())
