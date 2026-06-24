import math
from typing import List, Dict, Any

import torch
import torch.nn.functional as F

from .base import BaseScorer


def _positions_from_mask(mask: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Score token i using logits[i-1] (next-token prediction), so require i > 0.
    """
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    return idx[idx > 0].to(device)


def _pack(span_vals: List[float]) -> Dict[str, Any]:
    mean_val = float(sum(span_vals) / len(span_vals)) if span_vals else float("nan")
    return {"mean": mean_val, "spans": span_vals}


class AvgLogProbScorer(BaseScorer):
    name = "avg_logprob"
    aggregation = "whole"

    def score_from_logits(self, logits, input_ids, span_masks, gen_mask):
        idx = _positions_from_mask(gen_mask, logits.device)
        if idx.numel() == 0:
            return {"mean": float("nan"), "spans": []}

        pred = logits[idx - 1]
        tgt = input_ids[idx]
        logp = F.log_softmax(pred, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        val = float(logp.mean().item())
        return {"mean": val, "spans": []}


class DistributionalPerplexityScorer(BaseScorer):
    """
    dist_ppl = exp(mean_t H(p_t)) per span; then mean across spans.
    """
    name = "dist_ppl"
    aggregation = "spans_mean"

    def score_from_logits(self, logits, input_ids, span_masks, gen_mask):
        span_vals: List[float] = []
        for mask in span_masks:
            idx = _positions_from_mask(mask, logits.device)
            if idx.numel() == 0:
                continue

            pred = logits[idx - 1]
            logp = F.log_softmax(pred, dim=-1)
            p = logp.exp()
            entropy = -(p * logp).sum(dim=-1).mean()
            dist_ppl = torch.exp(entropy)
            span_vals.append(float(dist_ppl.item()))
        return _pack(span_vals)


class SelfCertaintyScorer(BaseScorer):
    """
    self_certainty = mean_t KL(p_t || Uniform) = mean_t (log(V) - H(p_t))
    per span; then mean across spans.
    """
    name = "self_certainty"
    aggregation = "spans_mean"

    def score_from_logits(self, logits, input_ids, span_masks, gen_mask):
        span_vals: List[float] = []
        V = logits.size(-1)
        logV = math.log(V)

        for mask in span_masks:
            idx = _positions_from_mask(mask, logits.device)
            if idx.numel() == 0:
                continue

            pred = logits[idx - 1]
            logp = F.log_softmax(pred, dim=-1)
            p = logp.exp()
            entropy = -(p * logp).sum(dim=-1).mean()
            sc = logV - entropy
            span_vals.append(float(sc.item()))
        return _pack(span_vals)