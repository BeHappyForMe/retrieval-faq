export TASK_NAME=faq

python faq.py \
    --model_type bert \
    --model_name_or_path /Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch \
    --config_name /Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch \
    --tokenizer_name /Users/zhoup/develop/NLPSpace/my-pre-models/chinese_wwm_pytorch \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ./data/preprocessed_synonymous.csv \
    --output_dir ./synonymous_model \
    --max_seq_length 256 \
    --per_gpu_eval_batch_size=16   \
    --per_gpu_train_batch_size=16 \
    --num_train_epochs 5 \
    --learning_rate 2e-5

notExce(){
export TASK_NAME=faq

python faq.py \
    --model_type bert \
    --model_name_or_path ./synonymous_model  \
    --config_name ./synonymous_model \
    --tokenizer_name ./synonymous_model \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --data_dir ./data/preprocessed_synonymous.csv \
    --output_dir ./synonymous_model \
    --max_seq_length 125 \
    --per_gpu_eval_batch_size=16
}