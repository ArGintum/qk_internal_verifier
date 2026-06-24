from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import torch

CHAT_PROMPT_TEMPLATE = """{problem}

Solution:
"""


@dataclass
class TokenizationResult:
    inputs: Dict[str, torch.Tensor]

    # problem span
    first_problem_tok_idx: int
    last_problem_tok_idx: int

    # solution span (everything after problem)
    first_think_tok_idx: Optional[int]
    last_think_tok_idx: Optional[int]

    first_post_think_tok_idx: Optional[int]

    seq_len: int
    offsets: List[Tuple[int, int]]
    full_text: str


def _char_span_to_token_span(
    offsets: List[Tuple[int, int]],
    start_char: int,
    end_char: int,
) -> Tuple[Optional[int], Optional[int]]:
    first_idx = None
    last_idx = None

    for i, (s, e) in enumerate(offsets):
        if e <= start_char:
            continue
        if s >= end_char:
            break
        if first_idx is None:
            first_idx = i
        last_idx = i

    return first_idx, last_idx


def special_tokenize(
    tokenizer,
    problem: str,
    output_text: str,
    device: torch.device,
) -> TokenizationResult:
    prompt_text = CHAT_PROMPT_TEMPLATE.format(problem=problem)
    full_text = prompt_text + output_text

    tok = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
        max_length=27000,
        truncation=True,
    )

    offsets = tok["offset_mapping"][0].tolist()
    input_ids = tok["input_ids"]
    seq_len = input_ids.shape[1]


    problem_start_char = 0
    problem_end_char = len(problem)

    first_problem_tok_idx, last_problem_tok_idx = _char_span_to_token_span(
        offsets, problem_start_char, problem_end_char
    )
    if first_problem_tok_idx is None:
        raise RuntimeError("Could not find problem span in tokenization.")

    solution_start_char = prompt_text.find("Solution:")
    if solution_start_char == -1:
        raise RuntimeError("Could not find 'Solution:' in prompt.")

    first_think_tok_idx = None
    for i, (_s, e) in enumerate(offsets):
        if e > solution_start_char:
            first_think_tok_idx = i
            break

    last_think_tok_idx = seq_len - 1 if first_think_tok_idx is not None else None

    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    print(seq_len)
    return TokenizationResult(
        inputs=inputs,
        first_problem_tok_idx=first_problem_tok_idx,
        last_problem_tok_idx=last_problem_tok_idx,
        first_think_tok_idx=first_think_tok_idx,
        last_think_tok_idx=last_think_tok_idx,
        first_post_think_tok_idx=None,
        seq_len=seq_len,
        offsets=offsets,
        full_text=full_text,
    )