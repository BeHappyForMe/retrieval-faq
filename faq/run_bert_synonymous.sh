export TASK_NAME=synonymous

python bert_synonymous.py \
    --model_type bert \
    --model_name_or_path /Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_predict \
    --do_lower_case \
    --data_dir ../data/ \
    --output_dir ./models/synonymous_model \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8 \
    --num_train_epochs 5 \
    --learning_rate 2e-5   \
    --loss_type  cosine