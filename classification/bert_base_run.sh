#!/usr/bin/env bash
export TASK_NAME=faq_bot_bert

python ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
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
    --eval_all_checkpoints  \
    --output_dir ./models/$TASK_NAME/