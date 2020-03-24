export TASK_NAME=faq

python bert_base.py \
    --model_type bert \
    --model_name_or_path /Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch \
    --task_name $TASK_NAME \
    --do_lower_case \
    --do_predict \
    --do_eval \
    --data_dir ../../data/baoxian_right.csv \
    --evaluate_dir ../../data/baoxian_evaluate.csv \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16   \
    --learning_rate 2e-5  \
    --output_type  avg