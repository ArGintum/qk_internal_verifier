model=openai/gpt-oss-20b
smodel=openai-gpt-oss-20b

for dataset in AIME_2025 AIME_2026 HMMT_FEB_2026 MATH500
do
	CUDA_VISIBLE_DEVICES=3 python collect_logits_stats.py --path ./outputs/$dataset-$smodel.jsonl --model $model --device cuda --save_json ./outputs/logits_$dataset-$smodel.json
done


