import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any
from typing import Iterable, Dict, Any, List, Tuple, Optional
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams

csv.field_size_limit(sys.maxsize)

import os
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
os.environ["SSL_CERT_FILE"] = "/etc/ssl/certs/ca-certificates.crt"

def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert competition mathematician. "
                "Solve the problem step by step. "
                "Then output only the final result as \\boxed{{}}. "
                "Do not include any other text."
            ),
        },
        {
            "role": "user",
            "content": problem,
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def extract_boxed_answer(text: str) -> str:
    # matches = re.findall(r"\\boxed\{(.*?)\}", text)
    #matches = re.findall(r"(?:boxed{)(.*)(?:})", text)
    matches = re.findall(r"\\boxed\{(.*?)\}", text)
    return matches[-1] if matches else ""


def sample_trajectories_with_retry(
    llm: LLM,
    prompt: str,
    num_samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    max_extra_attempts: int = 3,
) -> tuple[list[Dict[str, Any]], List[str]]:
    """
    Keep sampling until we have `num_samples` valid \\boxed{} answers
    or we hit `max_extra_attempts` rounds.
    """

    trajectories: List[Dict[str, Any]] = []
    valid_answers: List[str] = []

    remaining_needed = num_samples
    attempts = 0
    global_sample_index = 0

    while remaining_needed > 0 and attempts <= max_extra_attempts:
        print('attempts', attempts, 'max_extra_attempts', max_extra_attempts)
        attempts += 1

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            presence_penalty = presence_penalty,
            n=num_samples,
            stop=None,
        )


        print('START LLM GENERATE')
        outputs = llm.generate([prompt], sampling_params)
        print('END LLM GENERATE')
        output = outputs[0]

        for out in output.outputs:
            text = out.text
            parsed = extract_boxed_answer(text)
            has_boxed = bool(parsed)

            if has_boxed:
                trajectories.append(
                    {
                        "sample_index": global_sample_index,
                        "attempt_index": attempts,
                        "raw_text": text,
                        "parsed_answer": parsed,
                        "has_boxed": has_boxed,
                    }
                )
                global_sample_index += 1

                valid_answers.append(parsed)
                remaining_needed -= 1
                if remaining_needed == 0:
                    break

    return trajectories, valid_answers


def run_self_consistency(
    csv_path: str,
    output_path: str,
    model_name: str,
    num_samples: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5,
    max_tokens: int = 128,
    limit: int | None = None,
    tensor_parallel_size: int = 2,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.6,
    max_num_seqs: int = 32,
    max_extra_attempts: int = 3,
) -> None:

    csv_path = Path(csv_path)
    output_path = Path(output_path)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    problems: List[Dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "problem" not in reader.fieldnames or "answer" not in reader.fieldnames:
            raise ValueError("CSV must contain 'problem' and 'answer' columns.")

        for row in reader:
            problems.append(
                {
                    "problem": row["problem"],
                    "gold_answer": row["answer"],
                }
            )
            if limit is not None and len(problems) >= limit:
                break

    print(f"Loaded {len(problems)} problems.")

    out_f = output_path.open("w", encoding="utf-8")

    for idx, item in enumerate(problems):
        problem = item["problem"]
        gold_answer = item["gold_answer"]

        print(f"\n=== Problem {idx+1}/{len(problems)} ===")
        print(problem)

        prompt = build_prompt(problem, tokenizer)

        trajectories, valid_answers = sample_trajectories_with_retry(
            llm=llm,
            prompt=prompt,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            max_extra_attempts=max_extra_attempts,
        )

        record = {
            "problem_index": idx,
            "problem": problem,
            "gold_answer": gold_answer,
            "requested_num_samples": num_samples,
            "num_valid_samples": len(valid_answers),
            "valid_answers": valid_answers,
            "trajectories": trajectories,
        }

        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_f.flush()

    out_f.close()
    print(f"\nDone. Results written to: {output_path}")


def main():

    print('CUDA VISIBLE DEVICES', os.environ["CUDA_VISIBLE_DEVICES"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/AIME2025.csv")
    parser.add_argument("--out", default="outputs/AIME2025-Llama-3.1-8B-Instruct.jsonl")
    parser.add_argument("--model", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--num-samples", type=int, default=8)

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-extra-attempts", type=int, default=5)

    args = parser.parse_args()

    run_self_consistency(
        csv_path=args.csv,
        output_path=args.out,
        model_name=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        presence_penalty=args.presence_penalty,
        max_tokens=args.max_tokens,
        limit=args.limit,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_extra_attempts=args.max_extra_attempts,
    )


if __name__ == "__main__":
    main()
