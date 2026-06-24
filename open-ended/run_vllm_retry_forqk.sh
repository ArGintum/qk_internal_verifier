model=openai/gpt-oss-20b
smodel=openai-gpt-oss-20b

for dataset in AIME_2025 AIME_2026 HMMT_FEB_2026 MATH500
do
        CUDA_VISIBLE_DEVICES=1 python run_vllm_retry_forqk.py --csv data/$dataset.csv --out outputs/$dataset-$smodel.jsonl --model $model --gpu-memory-utilization 0.9 --tensor-parallel-size 1 --temperature=0.7 --top-p=0.8 --top-k=20 --presence_penalty=1.5 --max-tokens=7000 --max-model-len 8192
done

