python ./faq_customize.py \
     --train_file ./data/train.csv \
     --dev_file ./data/dev.csv \
     --output_dir ./model_pkl/customize/attention \
     --embed_size 512 \
     --head_attention_size 64 \
     --num_heads  8 \
     --num_epochs 5  \
     --loss_function  Hinge