export TASK_NAME=faq

python faq_predict.py \
    --model_type bert \
    --model_name_or_path /students/julyedu_563240/huggingface_transformers/chinese_wwm_pytorch \
    --config_name /students/julyedu_563240/huggingface_transformers/chinese_wwm_pytorch \
    --tokenizer_name /students/julyedu_563240/huggingface_transformers/chinese_wwm_pytorch \
    --task_name $TASK_NAME \
    --do_lower_case \
    --data_dir ../data/baoxianzhidao_filter.csv \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-5  \
    --output_type  pooled
