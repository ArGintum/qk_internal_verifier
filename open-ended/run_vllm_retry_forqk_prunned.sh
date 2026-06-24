model=Qwen-Qwen3-14B

for dataset in AIME_2025 AIME_2026 HMMT_FEB_2026 MATH500
do
	CUDA_VISIBLE_DEVICES=0 python run_vllm_retry_forqk.py --csv data/$dataset.csv --out "${model}_${dataset}_prunned.jsonl" --model "/workspace/reasoning_clean/${model}_${dataset}_zeroed_heads" --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --temperature=0.7 --top-p=0.8 --top-k=20 --presence_penalty=1.5 --max-tokens=7000 --max-model-len 8196 --max-extra-attempts 0;
	CUDA_VISIBLE_DEVICES=4 python run_vllm_retry_forqk.py --csv data/$dataset.csv --out "${model}_${dataset}_prunned_rnd.jsonl" --model "/workspace/reasoning_clean/${model}_${dataset}_zeroed_heads_rnd" --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --temperature=0.7 --top-p=0.8 --top-k=20 --presence_penalty=1.5 --max-tokens=7000 --max-model-len 8196 --max-extra-attempts 0;
done

