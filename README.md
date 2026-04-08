# Vinci Multi-System Orchestration

Contains the codebase for a multi-layered, multi-agent AI system.

## Features
- **Dynamic Routing**: Selects optimal models per domain (clinical vs general data).
- **Fallback Recovery**: Fails gracefully if OpenRouter/Gemini dies.
- **Clinical Safety Layer**: Prevents definitive medical diagnoses.
- **Medley-Style Consensus (New)**: Parallel dual-model generation with arbiter synthesis.
- **Isaree-Style Pipelines (New)**: Multi-step inference pipelines for structured evaluation.
- **Copilot Grounding (New)**: Real-time queries to PubMed & API RxNorm for factual context.
- **MedPerf Benchmarking (New)**: Autonomous MLCommons-style benchmarking and observability scoring.
- **Autonomous Intent Routing (Own AI)**: Evaluates user prompts to auto-select optimal pipelines.
- **Self-Reflective Loops (Own AI)**: Automatically attempts to self-correct low scoring or unsafe benchmark logs.

## Demo / Quickstart

Run the built-in demo to see the orchestration layer in action:

```bash
python demos.py
```

### Reviewing the Architecture
For an in-depth view of how the engine connects with the router and safety limits, read [architecture.md](architecture.md).
