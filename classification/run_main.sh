python ./main.py  \
    --train_file ./data/train.csv  \
    --dev_file  ./data/dev.csv  \
    --output_dir  ./models/main/crossent  \
    --num_epochs  5  \
    --vocab_size  50000  \
    --hidden_size  300  \
    --embed_size  300  \
    --batch_size  16   \
    --loss_function  CrossEntropy