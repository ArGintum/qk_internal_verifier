import argparse
import json
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from qk_utils.qk_hooks_model import QKHooksModel, get_qk_config
from qk_utils.tokenization_gpt_utils import special_tokenize
from qk_utils.qk_metrics import QKSpec, derive_qk_scores, SpanFilter, spec_base_name

import gc
import os 
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

SPAN_ALIASES = {"problem_span", "think_span", "post_think_span"}


def load_jsonl_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line.strip())


def build_specs_for_math_prompt():
    """
    Base specs + additional variants where spans are filtered to "\n\n".
    """
    base_specs = [
        QKSpec(q_alias="last_token", k_alias="problem_last", mode="token"),
    ]

    newline_filter = SpanFilter(kind="char_eq", value="\n\n")
    newline_specs = []

    for spec in base_specs:
        if spec.mode != "span":
            continue

        q_filter = newline_filter if spec.q_alias in SPAN_ALIASES else None
        k_filter = newline_filter if spec.k_alias in SPAN_ALIASES else None

        if q_filter is None and k_filter is None:
            continue

        newline_specs.append(
            QKSpec(
                q_alias=spec.q_alias,
                k_alias=spec.k_alias,
                mode="span",
                q_filter=q_filter,
                k_filter=k_filter,
            )
        )

    return base_specs + newline_specs


def process_generation(
    hooks_model: QKHooksModel,
    tokenizer,
    device: torch.device,
    problem: str,
    output_text: str,
    qk_config: Dict,
    specs: List[QKSpec],
) -> Dict[str, np.ndarray]:

    toks = special_tokenize(
        tokenizer=tokenizer,
        problem=problem,
        output_text=output_text,
        device=device,
    )

    with torch.no_grad():
        _, captured = hooks_model(toks.inputs)

    per_spec_scores = derive_qk_scores(
        hook_outputs=captured,
        token_positions=toks,
        qk_config=qk_config,
        specs=specs,
    )

    del toks
    del captured

    return per_spec_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="outputs/50prob_gpt-oss_v1_tools.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    if args.save_json is None:
        name = str(args.path).split("/")[-1].replace(".jsonl", "")
        model_name = args.model.split("/")[-1]
        args.save_json = f"stats/{name}_{model_name}-qk-stats.json"
        print(args.save_json)

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    max_memory = {
        0: "128GB",
        "cpu": "96GB"}

    hooks_model = QKHooksModel(args.model, args.device, max_memory = max_memory)
    model_config = hooks_model.pretrained.model.config
    qk_config = get_qk_config(model_config)

    specs = build_specs_for_math_prompt()

    data = list(load_jsonl_file(args.path))
    n_rows = len(data)
    print(f"Loaded {n_rows} rows from {args.path}")

    L = qk_config["L"]
    H = qk_config["H"]

    scores_dict: Dict[str, Dict[str, List[List[float]]]] = {}

    for spec in specs:
        base_name = spec_base_name(spec)
        scores_dict[base_name] = {}
        for l in range(L):
            for h in range(H):
                lh = f"{l}_{h}"
                scores_dict[base_name][lh] = []

    for ridx in tqdm(range(n_rows), desc="Processing rows"):
        gc.collect()
        torch.cuda.empty_cache()

        example = data[ridx]
        problem = example["problem"]
        generations = [
            t["raw_text"] for t in example["trajectories"] if t.get("has_boxed")
        ]

        if not generations:
            for base_name in scores_dict:
                for lh in scores_dict[base_name]:
                    scores_dict[base_name][lh].append([])
            continue

        row_values: Dict[str, Dict[str, List[float]]] = {
            base_name: {lh: [] for lh in scores_dict[base_name]}
            for base_name in scores_dict
        }

        for gen_text in generations:
            gen_text = str(gen_text)
            #print(gen_text)
            print('gen length', len(gen_text))

            spec_scores = process_generation(
                hooks_model=hooks_model,
                tokenizer=tokenizer,
                device=device,
                problem=problem,
                output_text=gen_text,
                qk_config=qk_config,
                specs=specs,
            )

            for base_name, arr in spec_scores.items():
                for l in range(L):
                    for h in range(H):
                        lh = f"{l}_{h}"
                        row_values[base_name][lh].append(float(arr[l, h]))

            del spec_scores

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for base_name in scores_dict:
            for lh in scores_dict[base_name]:
                scores_dict[base_name][lh].append(row_values[base_name][lh])

    for base_name, layer_head_dict in scores_dict.items():
        for lh, rows in layer_head_dict.items():
            if len(rows) != n_rows:
                raise RuntimeError(
                    f"{base_name}[{lh}] has {len(rows)} rows, expected {n_rows}"
                )

    save_path = args.save_json
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(scores_dict, f)

    print(f"Saved QK scores dict to {save_path}")


if __name__ == "__main__":
    main()
