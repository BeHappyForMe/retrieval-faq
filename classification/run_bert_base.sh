#!/usr/bin/env bash
export TASK_NAME=faq_bot_bert

python ./bert_base.py \
    --model_type bert \
    --model_name_or_path /Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./data/  \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps  5000  \
    --eval_all_checkpoints  \
    --output_dir ./models/$TASK_NAME/