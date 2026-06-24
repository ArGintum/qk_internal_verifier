from abc import ABC, abstractmethod
from typing import List, Dict, Any, Literal
import torch

Aggregation = Literal["whole", "spans_mean"]


class BaseScorer(ABC):
    name: str
    aggregation: Aggregation = "spans_mean"

    @abstractmethod
    def score_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        span_masks: List[torch.Tensor],
        gen_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        ...