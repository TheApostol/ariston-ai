Act as an AI workflow architect.

Convert static workflows into composable pipelines.

Goal:
Replace:
    if workflow == "diagnosis":

With:
    pipeline = [step1, step2, step3]

Each step should:
- take context
- return updated context

Implement:
- clinical reasoning pipeline
- regulatory analysis pipeline

Make it extensible for future agents.
