from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from collections import OrderedDict

from .tokenization_gpt_utils import TokenizationResult
from tqdm import tqdm

@dataclass
class SpanFilter:
    """
    Simple filter over token spans.

    kind:
      - "char_eq": keep only tokens whose underlying text == value
    """
    kind: str
    value: str


@dataclass
class QKSpec:
    """
    Describes a Q/K comparison to compute.

    mode:
        - "token": single query token vs single key token
        - "span": Q-span vs K-span; all pairwise dots, then L2 norm
    Optional filters further restrict which tokens from spans are used.
    """
    q_alias: str
    k_alias: str
    mode: str = "token"  # "token" or "span"
    q_filter: Optional[SpanFilter] = None
    k_filter: Optional[SpanFilter] = None


def angular_dist(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    return torch.sum(vec_a * vec_b)


def _alias_to_indices(alias: str, toks: TokenizationResult) -> np.ndarray:
    L = toks.seq_len

    # problem-related 
    if alias == "problem_last":
        return np.array([toks.last_problem_tok_idx])
    if alias == "problem_span":
        return np.arange(toks.first_problem_tok_idx, toks.last_problem_tok_idx + 1)

    # global last token
    if alias == "last_token":
        return np.array([L - 1])

    # think
    if alias == "think_span":
        if toks.first_think_tok_idx is None or toks.last_think_tok_idx is None:
            return np.array([], dtype=int)
        return np.arange(toks.first_think_tok_idx, toks.last_think_tok_idx + 1)

    # last token of think
    if alias == "post_think_last":  
        if toks.last_think_tok_idx is None:
            return np.array([], dtype=int)
        return np.array([toks.last_think_tok_idx])

    # post-think, after </think>
    if alias == "post_think_span":
        if toks.first_post_think_tok_idx is None:
            return np.array([], dtype=int)
        return np.arange(toks.first_post_think_tok_idx, L)

    # fallback
    return np.array([], dtype=int)


def _head_slice(
    tensor: torch.Tensor,  # [B, T, D]
    head_index: int,
    H_span: int,
) -> torch.Tensor:
    """
    Returns [T, head_dim] for a given head from [1, T, D].
    """
    assert tensor.ndim == 3 and tensor.shape[0] == 1
    start = head_index * H_span
    end = (head_index + 1) * H_span
    return tensor[0, :, start:end]


def _single_token_score(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    q_idx: int,
    k_idx: int,
    q_head: int,
    k_head: int,
    H_span: int,
) -> torch.Tensor:
    q_head_tensor = _head_slice(query_layer, q_head, H_span)
    k_head_tensor = _head_slice(key_layer, k_head, H_span)
    return angular_dist(q_head_tensor[q_idx], k_head_tensor[k_idx])


def _span_score(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    q_indices: Sequence[int],
    k_indices: Sequence[int],
    q_head: int,
    k_head: int,
    H_span: int,
) -> torch.Tensor:
    """
    Compute all pairwise dot products between Q-span and K-span tokens
    in the given heads via a batched matmul, then return L2 norm
    (Frobenius norm of the QK^T matrix).
    """
    if len(q_indices) == 0 or len(k_indices) == 0:
        return torch.tensor(float("nan"), dtype=torch.float32)

    
    q_head_tensor = _head_slice(query_layer, q_head, H_span).to(torch.float32)  # [T, d]
    k_head_tensor = _head_slice(key_layer, k_head, H_span).to(torch.float32)    # [T, d]

   
    q_vecs = q_head_tensor[q_indices]  # [Q, d]
    k_vecs = k_head_tensor[k_indices]  # [K, d]

  
    M = q_vecs @ k_vecs.T
    if not torch.isfinite(M).all():
        print(
            "Non-finite in M for head", q_head, "k_head", k_head,
            "q_len", len(q_indices), "k_len", len(k_indices)
        )
    sq_sum = torch.einsum("ij,ij->", M, M)
    if not torch.isfinite(sq_sum):
        print("Non-finite sq_sum in span_score:", sq_sum)

    return torch.sqrt(sq_sum)


def _apply_filter(
    indices: np.ndarray,
    span_filter: Optional[SpanFilter],
    toks: TokenizationResult,
) -> np.ndarray:
    """
    Given token indices and a SpanFilter, return filtered indices.
    """
    if span_filter is None or len(indices) == 0:
        return indices

    kind = span_filter.kind
    value = span_filter.value
    full_text = toks.full_text
    offsets = toks.offsets

    kept = []
    if kind == "char_eq":
        for i in indices:
            s, e = offsets[int(i)]
            if full_text[s:e] == value:
                kept.append(i)
    else:
        raise ValueError(f"Unknown SpanFilter kind: {kind}")

    return np.array(kept, dtype=int)


def _filter_suffix(span_filter: Optional[SpanFilter], prefix: str) -> str:
    if span_filter is None:
        return ""
    if span_filter.kind == "char_eq":
        v = span_filter.value.replace("\n", "\\n")  # make it readable in names
        return f"|{prefix}={v}"
    return f"|{prefix}=filtered"


def spec_base_name(spec: QKSpec) -> str:
    """
    Canonical name for a QKSpec, including filter info.
    Used consistently both in derive_qk_scores and in the main script.
    """
    k_suffix = _filter_suffix(spec.k_filter, "k")
    q_suffix = _filter_suffix(spec.q_filter, "q")
    return f"K:{spec.k_alias}{k_suffix};Q:{spec.q_alias}{q_suffix}"


def derive_qk_scores(
    hook_outputs: "OrderedDict[str, torch.Tensor]",
    token_positions: TokenizationResult,
    qk_config: Dict,
    specs: List[QKSpec],
) -> Dict[str, np.ndarray]:
    L = qk_config["L"]
    H = qk_config["H"]
    H_span = qk_config["H_span"]
    Q_per_K = qk_config["Q_per_K"]

    spec_scores: Dict[str, np.ndarray] = {}

    for spec in specs:
        base_name = spec_base_name(spec)

        vals = np.full((L, H), np.nan, dtype=np.float32)

        q_indices = _alias_to_indices(spec.q_alias, token_positions)
        k_indices = _alias_to_indices(spec.k_alias, token_positions)
        q_indices = _apply_filter(q_indices, spec.q_filter, token_positions)
        k_indices = _apply_filter(k_indices, spec.k_filter, token_positions)

        for l in range(L):
            q_layer = hook_outputs[f"query_vec_{l}"]
            k_layer = hook_outputs[f"key_vec_{l}"]

            for h in range(H):
                k_head = h // Q_per_K

                if spec.mode == "token":
                    if len(q_indices) == 0 or len(k_indices) == 0:
                        score = float("nan")
                    else:
                        q_idx = int(q_indices[-1])
                        k_idx = int(k_indices[-1])
                        score_tensor = _single_token_score(
                            q_layer, k_layer, q_idx, k_idx, h, k_head, H_span
                        )
                        score = float(score_tensor.item())

                elif spec.mode == "span":
                    score_tensor = _span_score(
                        q_layer,
                        k_layer,
                        q_indices,
                        k_indices,
                        h,
                        k_head,
                        H_span,
                    )
                    score = float(score_tensor.item())
                else:
                    raise ValueError(f"Unknown QKSpec mode: {spec.mode}")

                vals[l, h] = score

        spec_scores[base_name] = vals

    return spec_scores
