# Multi-System Orchestration Architecture

This document describes the orchestration architecture implemented to handle MAI-Dx, Medley, and MLCommons inspired workflows.

## System Components

### 1. Vinci Engine (`vinci_core/engine.py`)
The main entry point for all AI Requests. It is responsible for orchestrating the overall flow:
- Input Guardrails Validation
- Model routing and execution
- Model failover and fallback interception
- Output Guardrails Validation
- Normalizing results and packaging observability metadata

### 2. Model Router (`vinci_core/routing/model_router.py`)
Responsible for dynamically assigning instances of LLMs based on contextual needs:
- `layer="clinical"` -> Anthropic
- `layer="pharma"` -> Anthropic 
- `layer="data"` -> Gemini
- General requests -> OpenRouter
Also encapsulates the fallback rules (e.g. OpenRouter -> Gemini -> Anthropic).

### 3. Safety Guardrails (`vinci_core/safety/guardrails.py`)
Injects two major checks inspired by clinical-grade AI constraints:
- **Input Validation**: Discards inputs that are too short to generate a meaningful analysis, and assigns low confidence if structural complexity is missing.
- **Output Validation**: Blocks deterministic/diagnostic language often forbidden in medical settings (e.g., "you are suffering from").

## Observability & Metadata
Every `AIResponse` includes a robust `metadata` payload detailing:
- `latency_ms`: Total execution time.
- `provider`: Resolved execution model.
- `fallback_used`: Boolean indicating if a provider failed and was recovered.
- `failure_reason`: Error description if any provider failed.
- `confidence`: Generated score by safety constraints.
- `safety_flag`: Output state flag indicating safe or blocked output.
