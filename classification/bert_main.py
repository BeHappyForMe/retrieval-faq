import argparse
from collections import Counter
import code
import os
import logging
from tqdm import tqdm,trange
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,TensorDataset

from transformers import AdamW,get_linear_schedule_with_warmup
from transformers import BertConfig,BertForSequenceClassification,BertTokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.utils import DataProcessor,InputExample

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert":(BertConfig,BertForSequenceClassification,BertTokenizer)
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


class FAQProcessor(DataProcessor):
    def get_train_examples(self,data_dir):
        return self._create_examples(os.path.join(data_dir,"train.csv"),"train")

    def get_dev_examples(self,data_dir):
        return self._create_examples(os.path.join(data_dir,"dev.csv"),"dev")

    def get_labels(self):
        return [0,1]

    def _create_examples(self,path,set_type):
        df = pd.read_csv(path)
        examples = []
        titles = df["best_title"].tolist()
        reply = df["reply"].tolist()
        labels = df["is_best"].astype("int").tolist()
        for i in range(len(labels)):
            guid = "%s-%s" % (set_type,i)
            examples.append(InputExample(guid=guid,text_a=titles[i],text_b=reply[i],label=labels[i]))

def train(args,train_dataset,model,tokenizer):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size
    )
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) //args.gradient_accumulation_steps +1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    no_decay = ['bias','LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in  no_decay)],'weight_decay':args.weight_decay},
        {'params':[p for n,p in model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay':0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),desc="Epoch")
    set_seed(args)
    for _ in  train_iterator:
        epoch_iterator = tqdm(train_dataloader,desc="Iterator")
        for step,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attentiom_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}
            outputs = model(**inputs)
            # BertForSequenceClassification 若输入label则输出第一个为loss，否则第一个为logits
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps
            loss.backward()
            nn.utils.clip_grad_norm_(model.paratemers(),args.max_grad_norm)
            tr_loss += loss.item()
            if (step+1)%args.gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step+=1
            if args.max_steps > 0 and global_step>args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    return global_step,tr_loss/global_step

def evaluate(args, model, tokenizer):

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    total_counts = 0
    num_corrects = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[3]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2] if args.model_type in ['bert',
                                                                           'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            preds = torch.sigmoid(logits)
            preds[preds<0.5] = 0
            preds[preds>=0.5] = 1

        preds = preds.view(-1)
        lables = inputs['labels'].view(-1)
        num_corrects = torch.sum(preds==lables).item()
        total_counts += lables.shape[0]

    print("accuracy:{}".format(num_corrects / total_counts))
    return num_corrects / total_counts


def load_and_cache_examples(args,tokenizer,evaluate=False):
    processor = FAQProcessor()
    cached_feature_file = "cache_{}".format("dev" if evaluate else "train")
    if os.path.exists(cached_feature_file):
        features = torch.load(cached_feature_file)
    else:
        label_list = processor.get_labels()
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode="classification",
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=0
        )
        logger.info("saving features into cachefile %s",cached_feature_file)
        torch.save(features,cached_feature_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids,all_label)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--model_type',type=str,default=None,required=True,help="the model type")
    parser.add_argument('--model_name_or_path', type=str, default=None,required=True,
                        help=("Path to pre-trained model"))

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="directory containing the data")
    parser.add_argument("--output_dir", default="BERT_output", type=str, required=True,
                        help="The model output save dir")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--max_seq_length", default=128, type=int, required=False,
                        help="maximum sequence length for BERT sequence classificatio")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--do_lower_case",action='store_true')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument("--train_batch_size", default=32, type=int, required=False,
                        help="batch size for train and eval")
    parser.add_argument('--logging_steps', type=int, default=4000,
                        help="Log every X updates steps.")

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    processor = FAQProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else "./cache")
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else "./cache")
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else "./cache")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    model = model.to(args.device)

    # train
    if args.do_train:
        train_dataset = load_and_cache_examples(args,tokenizer,evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model = model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.output_dir)
        acc = evaluate(args,model,tokenizer)

if __name__ == '__main__':
    main()







