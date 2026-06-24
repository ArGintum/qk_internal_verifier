import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from logits_utils.tokenization import build_full_input_and_span_ranges, span_ranges_to_masks
from scorers.registry import build_scorers

os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

CHAT_PROMPT_TEMPLATE = """{problem}

Solution:
"""

def load_jsonl_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='outputs/50ref_Qwen3-4B-Thinking-2507_v1.jsonl')
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", type=str, default=None)

    parser.add_argument(
        "--methods",
        type=str,
        default="avg_logprob,dist_ppl,self_certainty",
        help="Comma-separated: ppl,dist_ppl,self_certainty,avg_logprob,entropy_mean",
    )
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--filter_boxed", action="store_true",
                        help="Keep only trajectories with has_boxed true.")
    parser.add_argument("--min_words_per_span", type=int, default=10,
                        help="If a newline chunk has < this many words, attach it to the next chunk.")

    args = parser.parse_args()

    if args.save_json is None:
        name = str(args.path).split("/")[-1].split(".")[0]
        args.save_json = f"stats/{name}_logit-stats.json"

    os.makedirs(os.path.dirname(args.save_json), exist_ok=True)

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
        device_map='auto',
    )
    # ILYA
    model = model.half()
    model.eval()
    model.config.use_cache = False  

    scorers = build_scorers(args.methods)
    print(f"Methods: {[s.name for s in scorers]}")

    data = list(load_jsonl_file(args.path))
    n_rows = len(data)
    print(f"Loaded {n_rows} rows from {args.path}")

    # scores_dict[method] = {"mean": rows->gens, "spans": rows->gens->spans}
    scores_dict: Dict[str, Dict[str, List]] = {
        s.name: {"mean": [], "spans": []} for s in scorers
    }

    for ridx in tqdm(range(n_rows), desc="Processing rows"):
        ex = data[ridx]
        problem = ex["problem"]
        prompt_text = CHAT_PROMPT_TEMPLATE.format(problem=problem)

        trajs = ex.get("trajectories", [])
        if args.filter_boxed:
            trajs = [t for t in trajs if t.get("has_boxed")]

        generations = [str(t.get("raw_text", "")) for t in trajs]
        if not generations:
            for s in scorers:
                scores_dict[s.name]["mean"].append([])
                scores_dict[s.name]["spans"].append([])
            continue

        row_mean: Dict[str, List[float]] = {s.name: [] for s in scorers}
        row_spans: Dict[str, List[List[float]]] = {s.name: [] for s in scorers}

        for gen_text in generations:
            enc, span_ranges, prompt_len = build_full_input_and_span_ranges(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                generation_text=gen_text,
                device=device,
                max_length=args.max_length,
                min_words_per_span=args.min_words_per_span,
                span_mode='newline'
            )

            T = enc["input_ids"].size(1)
            span_masks = span_ranges_to_masks(span_ranges, T=T, device='cpu')

            with torch.inference_mode():
                logits = model(**enc, use_cache=False).logits[0].to('cpu')  # [T, V]

            gen_mask = torch.zeros(T, dtype=torch.bool, device=logits.device)
            gen_mask[prompt_len:] = True

            input_ids_1d = enc["input_ids"][0].to(logits.device)  # [T]

            for s in scorers:
                res = s.score_from_logits(
                    logits=logits,
                    input_ids=input_ids_1d,
                    span_masks=span_masks,
                    gen_mask=gen_mask
                )
                row_mean[s.name].append(float(res["mean"]))
                row_spans[s.name].append([float(x) for x in res["spans"]])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del logits
            del gen_mask

        for s in scorers:
            scores_dict[s.name]["mean"].append(row_mean[s.name])
            scores_dict[s.name]["spans"].append(row_spans[s.name])

    # sanity check
    for name, obj in scores_dict.items():
        if len(obj["mean"]) != n_rows or len(obj["spans"]) != n_rows:
            raise RuntimeError(f"{name} has bad row count.")

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f)

    print(f"Saved logit scores dict to {args.save_json}")


if __name__ == "__main__":
    main()
