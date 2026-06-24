from __future__ import annotations
from typing import List, Tuple, Dict, Literal, Optional

import re
import torch
from transformers import PreTrainedTokenizerBase


def _count_words(s: str) -> int:
    return len([w for w in s.strip().split() if w])


def split_and_merge_by_newline(text: str, min_words: int = 10) -> List[str]:
    """
    Split by '\n'. If a chunk has < min_words, attach it to the NEXT chunk.
    Trailing short chunk attaches to previous.
    Empty lines are dropped.
    """
    raw = [x for x in text.split("\n") if x.strip() != ""]
    if not raw:
        return []

    merged: List[str] = []
    pending = ""

    i = 0
    while i < len(raw):
        chunk = raw[i].strip()

        if pending:
            chunk = pending + "\n" + chunk
            pending = ""

        if _count_words(chunk) < min_words and i < len(raw) - 1:
            pending = chunk
            i += 1
            continue

        merged.append(chunk)
        i += 1

    if pending:
        if merged:
            merged[-1] = merged[-1] + "\n" + pending
        else:
            merged = [pending]

    return merged



_ABBREV = {
    "e.g.", "i.e.", "etc.", "vs.", "cf.", "approx.", "al.", "fig.", "no.", "dr.", "mr.", "ms.", "mrs.",
    "eq.", "eqs.", "thm.", "prop.", "lem.", "cor.", "def.", "ex.", "ref.", "sec.", "ch.",
}

_SENT_START_RE = re.compile(r"^(?:[A-Z]|\$|\\|\(|\[|\{|\d+\.|\d+\)|-|\*)")


def _looks_like_abbrev(token_with_period: str) -> bool:
    t = token_with_period.lower()
    if t in _ABBREV:
        return True
    if len(t) == 2 and t[0].isalpha() and t[1] == ".":
        return True
    return False


def _is_enumeration_marker(prev_token: str) -> bool:
    # Prevent splitting after "1." or "12." which would create a useless "1." sentence.
    return bool(re.fullmatch(r"\d+\.", prev_token))


def split_into_sentences_heuristic(text: str) -> List[str]:
    """
    Heuristic sentence splitter designed for math-solution style generations.

    Handles:
      - decimals (mostly safe because there's no whitespace after the dot)
      - abbreviations like e.g., i.e., Eq., Thm.
      - enumerated list markers "1." / "2."
      - newlines (treated as whitespace, not hard boundaries)

    Returns a list of sentence-ish strings (not merged by min_words yet).
    """
    s = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    s = re.sub(r"\n+", " ", s).strip()
    if not s:
        return []

    out: List[str] = []
    start = 0
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]
        if ch in ".!?":
            
            j = i
            while j + 1 < n and s[j + 1] == "." and ch == ".":
                j += 1

            
            end_punct = j
            
            k = end_punct - 1
            while k >= 0 and s[k] == " ":
                k -= 1
            token_end = k
            while k >= 0 and s[k] not in " ":
                k -= 1
            prev_token = s[k + 1 : token_end + 1] 
            prev_token_with_period = s[k + 1 : end_punct + 1] 

        
            should_split = True


            if ch == "." and _looks_like_abbrev(prev_token_with_period):
                should_split = False

            if ch == "." and _is_enumeration_marker(prev_token_with_period):
                should_split = False

        
            t = end_punct + 1
            if t < n and s[t] != " ":
                should_split = False
            while t < n and s[t] == " ":
                t += 1
            if t >= n:
                should_split = True
            else:
                if not _SENT_START_RE.match(s[t:]):
                    should_split = False

            if should_split:
                sent = s[start : end_punct + 1].strip()
                if sent:
                    out.append(sent)
                start = t 
                i = t
                continue

            i = end_punct + 1
            continue

        i += 1

    tail = s[start:].strip()
    if tail:
        out.append(tail)

    return out


def split_and_merge_by_sentence(text: str, min_words: int = 10) -> List[str]:
    """
    Sentence split + merge small sentences into the NEXT sentence (like newline version).
    Trailing short chunk attaches to previous.
    """
    raw = [x.strip() for x in split_into_sentences_heuristic(text) if x.strip()]
    if not raw:
        return []

    merged: List[str] = []
    pending = ""

    i = 0
    while i < len(raw):
        chunk = raw[i]

        if pending:
            chunk = pending + " " + chunk
            pending = ""

        if _count_words(chunk) < min_words and i < len(raw) - 1:
            pending = chunk
            i += 1
            continue

        merged.append(chunk)
        i += 1

    if pending:
        if merged:
            merged[-1] = merged[-1] + " " + pending
        else:
            merged = [pending]

    return merged


SpanMode = Literal["newline", "sentence"]


def split_text_into_spans(text: str, min_words: int, mode: SpanMode) -> List[str]:
    if mode == "newline":
        return split_and_merge_by_newline(text, min_words=min_words)
    if mode == "sentence":
        return split_and_merge_by_sentence(text, min_words=min_words)
    raise ValueError(f"Unknown span mode: {mode}")


def build_full_input_and_span_ranges(
    tokenizer: PreTrainedTokenizerBase,
    prompt_text: str,
    generation_text: str,
    device: torch.device,
    max_length: int = 8192,
    min_words_per_span: int = 10,
    span_mode: SpanMode = "sentence",
) -> Tuple[Dict[str, torch.Tensor], List[Tuple[int, int]], int]:
    """
    Tokenize (prompt_text + generation_text) and return:
      enc: dict for model(**enc), batch=1
      span_ranges: list of (start, end) token indices in [0,T), generation-only
      prompt_len: prompt token length after truncation adjustment
    """
    
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt_text + generation_text, add_special_tokens=False).input_ids

    
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        gen_ids = tokenizer(generation_text, add_special_tokens=False).input_ids
        full_ids = prompt_ids + gen_ids

    
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        prompt_ids = prompt_ids[max(0, overflow):]

    prompt_len = len(prompt_ids)
    T = len(full_ids)

    input_ids = torch.tensor(full_ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
    enc = {"input_ids": input_ids, "attention_mask": attention_mask}

    chunks = split_text_into_spans(
        generation_text,
        min_words=min_words_per_span,
        mode=span_mode,
    )

    span_ranges: List[Tuple[int, int]] = []
    offset = prompt_len
    for ch in chunks:
        ch_ids = tokenizer(ch, add_special_tokens=False).input_ids
        if not ch_ids:
            continue

        start = offset
        end = min(offset + len(ch_ids), T)
        offset = end

        if start < end:
            span_ranges.append((start, end))

    if not span_ranges and prompt_len < T:
        span_ranges = [(prompt_len, T)]

    return enc, span_ranges, prompt_len


def span_ranges_to_masks(
    span_ranges: List[Tuple[int, int]],
    T: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Convert (start,end) ranges to boolean masks [T] for compatibility with existing scorer code.
    """
    masks: List[torch.Tensor] = []
    for (s, e) in span_ranges:
        m = torch.zeros(T, dtype=torch.bool, device=device)
        m[s:e] = True
        masks.append(m)
    return masks