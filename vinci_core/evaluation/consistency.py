"""
Consistency scoring — MedPerf pillar for output stability.

run_consistency_check(prompt, n=3) runs the same prompt through the engine
n times and measures variance across responses.  A low variance score
indicates the model produces stable, reproducible outputs for a given input.

Metrics returned:
  - consistency_score:  1.0 − normalised_variance  (higher = more consistent)
  - mean_length:        average response character length across runs
  - length_variance:    variance in response lengths (a simple proxy)
  - jaccard_similarity: pairwise token overlap averaged across all pairs
  - runs:               number of successful completions
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ariston.evaluation.consistency")


def _tokenize(text: str) -> set:
    """Lower-case word tokens, punctuation stripped."""
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def _pairwise_jaccard(responses: List[str]) -> float:
    """Average Jaccard similarity across all unique pairs of responses."""
    if len(responses) < 2:
        return 1.0
    pairs = [
        (responses[i], responses[j])
        for i in range(len(responses))
        for j in range(i + 1, len(responses))
    ]
    scores = []
    for a, b in pairs:
        ta, tb = _tokenize(a), _tokenize(b)
        union = ta | tb
        if not union:
            scores.append(1.0)
        else:
            scores.append(len(ta & tb) / len(union))
    return round(sum(scores) / len(scores), 4)


def _length_variance(lengths: List[int]) -> float:
    if len(lengths) < 2:
        return 0.0
    mean = sum(lengths) / len(lengths)
    return round(sum((x - mean) ** 2 for x in lengths) / len(lengths), 2)


async def run_consistency_check(
    prompt: str,
    n: int = 3,
    engine=None,
    layer: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run *prompt* through the engine *n* times and compute consistency metrics.

    Parameters
    ----------
    prompt:
        The clinical/regulatory prompt to test.
    n:
        Number of independent runs (default 3, as per MedPerf spec).
    engine:
        The Engine instance to use.  Defaults to the global ``engine`` singleton
        from ``vinci_core.engine`` (imported lazily to avoid circular imports).
    layer:
        Optional layer override passed to every engine call.
    model:
        Optional model override passed to every engine call.

    Returns
    -------
    dict with keys: consistency_score, jaccard_similarity, mean_length,
                    length_variance, runs, errors
    """
    if engine is None:
        from vinci_core.engine import engine as _engine  # lazy import
        engine = _engine

    responses: List[str] = []
    errors = 0

    for i in range(n):
        try:
            result = await engine.run(
                prompt=prompt,
                layer=layer,
                model=model,
                use_rag=False,  # disable RAG for deterministic comparison
            )
            responses.append(result.content or "")
        except Exception as exc:
            logger.warning(
                '{"event":"consistency_run_failed","run":%d,"error":"%s"}',
                i, type(exc).__name__,
            )
            errors += 1

    if not responses:
        logger.error('{"event":"consistency_check_no_responses","prompt_len":%d}', len(prompt))
        return {
            "consistency_score": 0.0,
            "jaccard_similarity": 0.0,
            "mean_length": 0,
            "length_variance": 0.0,
            "runs": 0,
            "errors": errors,
        }

    lengths = [len(r) for r in responses]
    mean_length = round(sum(lengths) / len(lengths))
    lv = _length_variance(lengths)
    jaccard = _pairwise_jaccard(responses)

    # Normalise length variance to [0, 1] using a soft cap of 50 000 chars²
    _MAX_LV = 50_000.0
    normalised_lv = min(lv / _MAX_LV, 1.0)

    # Blend length stability and token overlap into a single score
    consistency_score = round(0.5 * (1.0 - normalised_lv) + 0.5 * jaccard, 4)

    metrics: Dict[str, Any] = {
        "consistency_score": consistency_score,
        "jaccard_similarity": jaccard,
        "mean_length": mean_length,
        "length_variance": lv,
        "runs": len(responses),
        "errors": errors,
    }

    log_level = logger.warning if consistency_score < 0.70 else logger.info
    log_level(
        '{"event":"consistency_result","consistency_score":%.4f,"jaccard":%.4f,"runs":%d}',
        consistency_score, jaccard, len(responses),
    )

    return metrics
