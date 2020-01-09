# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

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
    BertConfig,
    BertModel,
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

logger = logging.getLogger(__name__)
# 这里选择BertModel，使用bert做一个简单的特征提取器
MODEL_CLASS = {"bert":(BertConfig,BertModel,BertTokenizer)}

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

    def get_candidates(self,file_dir):
        train_df = pd.read_csv(file_dir)
        candidates = train_df[train_df["is_best"]==1]
        best_title = candidates.apply(
            lambda row : row["question"] if row["question"] is not None and len(str(row["question"]))
            > len(str(row["title"])) else row["title"],
            axis=1
        )
        self.candidate_title = best_title.tolist()
        self.candidate_reply = candidates["reply"].tolist()
        return self._create_examples(self.candidate_title,"train")

    def _create_examples(self,lines,set_type):
        examples = []
        for (i,line) in enumerate(lines):
            guid = "%s-%s" % (set_type,i)
            examples.append(InputExample(guid=guid,text_a=line,text_b=None,label=1))
        return examples

def evaluate(args,model,dataset):
    results = []
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1,args.n_gpu)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,sampler=sampler,batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    for batch in tqdm(dataloader,desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids":batch[0],"attention_mask":batch[1],"token_type_ids":batch[2]}
            outputs = model(**inputs)
            sequence_output,pooled_output = outputs[:2]
            if args.output_type == "pooled":
                results.append(pooled_output)
            elif args.output_type == "avg":
                results.append(sequence_output.mean(1))
    return results


def load_examples(args,tokenizer):
    processor = FAQProcessor()
    # Load data features from cache or dataset file
    cached_features_file = "cached_{}_{}_{}_{}".format(
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(args.task_name),
            "features",
        )
    if os.path.exists(cached_features_file):
        features = torch.load(cached_features_file)
    else:
        examples = processor.get_candidates(args.data_dir)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=[1],
            output_mode="classification",
            max_length=args.max_seq_length,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        logger.info("saving features into cachefile %s",cached_features_file)
        torch.save(features,cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features],dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features],dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features],dtype=torch.long)
    dataset = TensorDataset(all_input_ids,all_attention_mask,all_token_type_ids)
    return dataset,processor.candidate_title,processor.candidate_reply

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
        help="Model type selected in the list:",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--output_type",
        default="pooled",
        type=str,
        required=False,
        choices=["pooled", "avg"],
        help="the type of choice output vector",
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
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
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
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

    set_seed(args)
    args.task_name = args.task_name.lower()
    args.model_type = args.model_type.lower()

    config_class,model_class,tokenizer_class = MODEL_CLASS[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        finetuning_task = args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else "./cache"
    )
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                cache_dir=args.cache_dir if args.cache_dir else "./cache")
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool('.ckpt' in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else "./cache"
    )
    model = model.to(device)

    if not os.path.exists("embeddings.pkl"):
        eval_dataset, candidate_title, candidate_reply = load_examples(args, tokenizer)
        outputs = evaluate(args, model, eval_dataset)
        candidate_embeddings = torch.cat([o.cpu().data for o in outputs]).numpy()
        torch.save([candidate_title,candidate_reply,candidate_embeddings],"embeddings.pkl")
    else:
        candidate_title, candidate_reply, candidate_embeddings = torch.load("embeddings.pkl")

    while True:
        question = input("你的问题是?\n")
        if len(str(question)) == 0:
            continue

        examples = [InputExample(guid=0,text_a=question,text_b=None,label=1)]
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
        all_input_ids = torch.tensor([f.input_ids for f in features],dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids,all_attention_masks,all_token_type_ids)
        outputs = evaluate(args,model,dataset)

        question_embedding = torch.cat([o.cpu().data for o in outputs]).numpy()
        scores = cosine_similarity(question_embedding,candidate_embeddings)[0]
        top5 = scores.argsort()[-5:][::-1]
        for index in top5:
            print("可能得答案，参考问题为:{},答案:{},得分:{}".format(candidate_title[index],candidate_reply[index],str(scores[index])))
            print()


if __name__ == '__main__':
    main()




