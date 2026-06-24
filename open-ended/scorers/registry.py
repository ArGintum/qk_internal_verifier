from .logit_metrics import (
    AvgLogProbScorer,
    DistributionalPerplexityScorer,
    SelfCertaintyScorer,
)

_NAME_TO_SCORER = {
    "avg_logprob": AvgLogProbScorer,
    "dist_ppl": DistributionalPerplexityScorer,
    "self_certainty": SelfCertaintyScorer,
}


def build_scorers(methods_csv: str):
    names = [m.strip() for m in methods_csv.split(",") if m.strip()]
    scorers = []
    for n in names:
        if n not in _NAME_TO_SCORER:
            raise ValueError(
                f"Unknown method '{n}'. Available: {sorted(_NAME_TO_SCORER.keys())}"
            )
        scorers.append(_NAME_TO_SCORER[n]())
    return scorers