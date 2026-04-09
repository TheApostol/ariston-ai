Act as an AI infrastructure engineer improving the model execution layer.

Tasks:
- Ensure all providers (Anthropic, Gemini, OpenRouter) return:
  { model, content, usage, metadata }
- Add timeout handling
- Add retry logic
- Improve error messages
- Normalize token usage across providers

Refactor this file to:
- Remove duplication
- Improve reliability
- Add observability hooks

Ensure compatibility with VinciEngine.
