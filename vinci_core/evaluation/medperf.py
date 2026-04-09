"""
MedPerf / MLCommons evaluation runner.
Runs engine responses against standard medical QA benchmarks.
Outputs scores compatible with MLCommons submission format.

Supported benchmarks:
  - MedQA (USMLE-style, 4-option MCQ)
  - PubMedQA (yes/no/maybe on abstracts)
  - MedMCQA (Indian medical entrance, 4-option)

Usage:
  python -m vinci_core.evaluation.medperf --benchmark medqa --n 50
"""

import json
import asyncio
import argparse
import os
from datetime import datetime, timezone
from typing import List, Dict

# Public benchmark data URLs (free, no registration)
BENCHMARK_URLS = {
    "medqa": "https://raw.githubusercontent.com/jind11/MedQA/master/data_clean/questions/US/test.jsonl",
    "pubmedqa": "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_set.json",
}

LOG_DIR = "benchmarks"


async def run_benchmark(benchmark: str = "medqa", n: int = 20) -> Dict:
    """
    Run n questions from a benchmark against the Ariston engine.
    Returns accuracy, per-question results, and MLCommons-format summary.
    """
    from vinci_core.engine import engine
    import httpx

    os.makedirs(LOG_DIR, exist_ok=True)

    url = BENCHMARK_URLS.get(benchmark)
    if not url:
        return {"error": f"Unknown benchmark: {benchmark}"}

    # Fetch benchmark data
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url)

    if benchmark == "medqa":
        questions = [json.loads(line) for line in r.text.strip().split("\n") if line][:n]
        results = await _run_medqa(questions, engine)
    elif benchmark == "pubmedqa":
        data = r.json()
        questions = list(data.items())[:n]
        results = await _run_pubmedqa(questions, engine)
    else:
        return {"error": "Unsupported benchmark"}

    correct = sum(1 for r in results if r.get("correct"))
    accuracy = correct / len(results) if results else 0

    summary = {
        "benchmark": benchmark,
        "model": "ariston-ai/claude-sonnet-4-6",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": len(results),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "results": results,
    }

    # Save MLCommons-format log
    log_path = os.path.join(LOG_DIR, f"medperf_{benchmark}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[MedPerf] {benchmark}: {correct}/{len(results)} = {accuracy:.1%} | saved to {log_path}")
    return summary


async def _run_medqa(questions: List[Dict], engine) -> List[Dict]:
    results = []
    for q in questions:
        prompt = _format_medqa_prompt(q)
        response = await engine.run(prompt=prompt, layer="clinical", use_rag=False)
        predicted = _extract_choice(response.content)
        correct_answer = q.get("answer_idx", q.get("answer", ""))
        results.append({
            "question": q.get("question", "")[:100],
            "predicted": predicted,
            "correct": correct_answer,
            "match": predicted.upper() == str(correct_answer).upper(),
            "correct": predicted.upper() == str(correct_answer).upper(),
        })
    return results


async def _run_pubmedqa(questions: List, engine) -> List[Dict]:
    results = []
    for qid, q in questions:
        context = " ".join(q.get("CONTEXTS", []))[:800]
        question = q.get("QUESTION", "")
        prompt = (
            f"Based on the following abstract, answer yes, no, or maybe.\n\n"
            f"Abstract: {context}\n\nQuestion: {question}\n\nAnswer (yes/no/maybe only):"
        )
        response = await engine.run(prompt=prompt, layer="clinical", use_rag=False)
        predicted = _extract_yesno(response.content)
        correct = q.get("final_decision", "")
        results.append({
            "qid": qid,
            "predicted": predicted,
            "correct": correct,
            "match": predicted.lower() == correct.lower(),
        })
    return results


def _format_medqa_prompt(q: Dict) -> str:
    options = q.get("options", {})
    opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())
    return (
        f"Question: {q.get('question', '')}\n\n"
        f"Options:\n{opts_text}\n\n"
        "Answer with the letter only (A, B, C, or D)."
    )


def _extract_choice(content: str) -> str:
    import re
    match = re.search(r'\b([A-Da-d])\b', content.strip()[:20])
    return match.group(1).upper() if match else content.strip()[0].upper()


def _extract_yesno(content: str) -> str:
    lower = content.lower().strip()
    if lower.startswith("yes"):
        return "yes"
    if lower.startswith("no"):
        return "no"
    return "maybe"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="medqa", choices=["medqa", "pubmedqa"])
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.benchmark, args.n))
