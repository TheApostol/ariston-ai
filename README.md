# Vinci Multi-System Orchestration

Contains the codebase for a multi-layered, multi-agent AI system.

## Features
- **Dynamic Routing**: Selects optimal models per domain (clinical vs general data).
- **Fallback Recovery**: Fails gracefully if OpenRouter/Gemini dies.
- **Clinical Safety Layer**: Prevents definitive medical diagnoses.

## Demo / Quickstart

Run the built-in demo to see the orchestration layer in action:

```bash
python demos.py
```

### Reviewing the Architecture
For an in-depth view of how the engine connects with the router and safety limits, read [architecture.md](architecture.md).
