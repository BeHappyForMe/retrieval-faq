import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertModel,
    BertPreTrainedModel,
    BertTokenizer,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from transformers.data.processors.utils import InputExample, DataProcessor

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

'''
    bert同义句训练模型
'''

logger = logging.getLogger(__name__)
model_path = "/share/data/lang/users/zeweichu/projs/qa-chinese/chinese_wwm_ext_pytorch"
MODEL_CLASS = {"bert":(BertConfig,BertTokenizer)}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class FAQProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )


    def prepare_data(self,file_dir):
        train_df = pd.read_csv(file_dir,sep='\t')
        self.candidate_title = train_df['best_title'].astype("str").tolist()
        self.candidate_reply = train_df['reply'].astype("str").tolist()
        self.candidate_translated = train_df['translated'].astype("str").tolist()

    def get_train(self):
        return self._create_pos_examples(self.candidate_title,self.candidate_translated,"pos"),\
               self._create_neg_examples(self.candidate_title,"neg")

    def _create_pos_examples(self,lines_a,lines_b,set_type):
        """创建同义句pair"""
        examples = []
        for (i,(line_a,line_b)) in enumerate(zip(lines_a,lines_b)):
            guid = "%s_%s" % (set_type,i)
            examples.append(InputExample(guid=guid,text_a=line_a,text_b=line_b,label=1))
        return examples

    def _create_neg_examples(self,lines,set_type):
        """创建相似度很高的非同义句pair"""
        examples = []
        for (i,line) in enumerate(lines):
            guid = "%s_%s" % (set_type,i)
            #TODO 这里这是做了简单抽样，应该修改
            line_b = random.sample(lines,1)[0]
            while line_b == line:
                line_b = random.sample(lines,1)[0]
            examples.append(InputExample(guid=guid,text_a=line,text_b=line_b,label=1))
        return examples


class BertForFAQHinge(BertPreTrainedModel):
    r"""
        该方法从BertForSequenceClassification稍加修改而已，
        损失函数采用hingeloss
    """

    def __init__(self, config):
        super(BertForFAQHinge, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.scoring = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        score = self.scoring(pooled_output)

        outputs = (score,) + outputs[2:]  # add hidden states and attention if they are here

        return outputs  # score, (hidden_states), (attentions)


def train(args, train_dataset, model, processor, tokenizer):
    """ train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

        # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    t_total = len(train_dataset) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    epochs_trained = 0
    global_step = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(
        epochs_trained,int(args.num_train_epochs),desc="Epoch",disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)
    for k in train_iterator:
        if k!=0:
            dataset,_,_=load_examples(args,args.task_name,tokenizer,processor)
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset,sampler=train_sampler,batch_size=args.train_batch_size)
        epoch_iterator = tqdm(train_dataloader,desc="Iteration",disable=args.local_rank not in [-1, 0])
        for step , batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            pos_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            neg_inputs = {"input_ids": batch[3], "attention_mask": batch[4], "token_type_ids": batch[5]}
            pos_outputs = model(**pos_inputs)
            neg_outputs = model(**neg_inputs)

            pos_score = pos_outputs[0]
            neg_score = neg_outputs[0]
            #计算hingeloss
            loss = args.margin + neg_score - pos_score
            loss[loss<0] = 0
            loss = torch.mean(loss)
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
                model.zero_grad()
                optimizer.step()
                scheduler.step()
                global_step+=1
        logger.info("average loss: " + str(tr_loss / global_step))
    return global_step,tr_loss/global_step


def evaluate(args, model, eval_dataset):
    # 把所有的问题全部用BERT编码
    scores = []

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            outputs = model(**inputs)

            score = outputs[0]
            scores.append(score)

    scores = [score.data.cpu().view(-1) for score in scores]
    scores = torch.cat(scores)

    return scores.numpy()


def load_examples(args,task,tokenizer,processor):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    pos_data,neg_data = processor.get_train()
    # Load data features from cache or dataset file
    logger.info("Creating features from dataset file at %s", args.data_dir)
    pos_features = convert_examples_to_features(
        pos_data,
        tokenizer,
        label_list=[1],
        output_mode="classification",
        max_length=args.max_seq_length,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )
    neg_features = convert_examples_to_features(
        neg_data,
        tokenizer,
        label_list=[1],
        output_mode="classification",
        max_length=args.max_seq_length,
        pad_on_left=bool(args.model_type in ["xlnet"]),
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    pos_input_ids = torch.tensor([f.input_ids for f in pos_features], dtype=torch.long)
    pos_attention_mask = torch.tensor([f.attention_mask for f in pos_features], dtype=torch.long)
    pos_token_type_ids = torch.tensor([f.token_type_ids for f in pos_features], dtype=torch.long)
    neg_input_ids = torch.tensor([f.input_ids for f in neg_features], dtype=torch.long)
    neg_attention_mask = torch.tensor([f.attention_mask for f in neg_features], dtype=torch.long)
    neg_token_type_ids = torch.tensor([f.token_type_ids for f in neg_features], dtype=torch.long)
    dataset = TensorDataset(pos_input_ids, pos_attention_mask, pos_token_type_ids, neg_input_ids,
                                neg_attention_mask, neg_token_type_ids)

    return dataset, processor.candidate_title, processor.candidate_reply


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--margin", default=5, type=float, help="The margin of hinge loss.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=2000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, tokenizer_class = MODEL_CLASS[args.model_type]
    model_class = BertForFAQHinge
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else "./synonymous_model/cache",
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else "./synonymous_model/cache",
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else "./synonymous_model/cache",
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    processor = FAQProcessor()
    processor.prepare_data(args.data_dir)
    dataset, candidate_title, candidate_reply = load_examples(args, args.task_name, tokenizer, processor)
    if args.do_train:
        train(args, dataset, model, processor, tokenizer)
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    if args.do_eval:
        while True:
            title = input("你的问题是？\n")
            if len(title.strip()) == 0:
                continue

            #将输入问题与所有训练数据的title组成句子pair，分别计算出score
            examples = [InputExample(guid=0, text_a=title, text_b=c, label=1) for c in candidate_title]
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=[1],
                output_mode="classification",
                max_length=args.max_seq_length,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )

            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

            dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
            scores = evaluate(args, model, dataset)
            top5_indices = scores.argsort()[-5:][::-1]
            for index in top5_indices:
                print("可能的答案，参考问题：" + candidate_title[index] + "\t答案：" + candidate_reply[index] + "\t得分：" + str(
                    scores[index]))
                print()


if __name__ == "__main__":
    main()
