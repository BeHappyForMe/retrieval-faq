import flask
from flask import Flask, render_template, session, request, redirect, url_for, jsonify
# from app import app
import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import AdamW
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer, BertModel
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features

import pandas as pd
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertModel.from_pretrained('models/synonymous_model/checkpoint-200000')
model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained('models/synonymous_model/checkpoint-200000')


class FaqProcessor(DataProcessor):
    def __init__(self, file_dir):
        train_df = pd.read_csv(file_dir + 'touzi_preprocessed_synonymous.csv', sep='\t')
        self.candidate_title = train_df['best_title'].astype("str").tolist()
        self.candidate_reply = train_df['reply'].astype("str").tolist()
        self.candidate_translated = train_df['translated'].astype("str").tolist()

    def _create_one_example(self, title):
        examples = [InputExample(guid=0, text_a=title, text_b=None, label=1)]
        return examples

    def prepare_replies(self, path):
        df = pd.read_csv(path)
        df = df.fillna(0)
        replies = [str(t) for t in df['reply'].tolist()]
        return replies


processor = FaqProcessor('./data/')


def get_scores(dataset):
    # 输出score
    scores = []

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=8)

    # 确保原问答数据集已经保存
    cached_predict_original_embeddings = os.path.join(
        './data/',
        "cached_{}_{}_{}".format(
            "pred",
            'original',
            'embeddings',
        ),
    )
    logger.info("Loading pred original embeddings from cached file %s", cached_predict_original_embeddings)
    original_embeddings = torch.load(cached_predict_original_embeddings)

    eval_question_embeddings = []
    for batch in tqdm(eval_dataloader, desc="predicting"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            eval_question_inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            eval_questions_outputs = model(**eval_question_inputs)[0].mean(1)
            eval_question_embeddings.append(eval_questions_outputs)
    eval_question_embeddings = torch.cat([o.cpu().data for o in eval_question_embeddings]).numpy()
    scores = cosine_similarity(eval_question_embeddings, original_embeddings)[0]
    return scores


def predict(title):
    examples = processor._create_one_example(title)
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=[0, 1],
        max_length=256,
        output_mode='classification',
        pad_on_left=False,  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)

    result = ''
    scores = get_scores(dataset)
    top5_indices = scores.argsort()[-5:][::-1]
    for i, index in enumerate(top5_indices):
        print("可能的答案，参考问题：" + processor.candidate_title[index] + "\t答案：" + processor.candidate_reply[
            index] + "\t得分：" + str(
            scores[index]))
        print()
        # 页面只输出概率最大的一个
        if i == 0:
            result += "最可能的答案，参考问题：" + processor.candidate_title[index] + "\t答案：" + processor.candidate_reply[
                index] + "\t得分：" + str(
                scores[index]) + '\n'

    return result


app = Flask(__name__)

app.secret_key = 'F12Zr47j\3yX R~X@H!jLwf/T'


@app.route("/")
def hello_world():
    return render_template('FQA.html')


@app.route('/fqa', methods=['GET'])
def fqa():
    while request.args.get('title'):
        title = str(request.args.get('title'))
        print(title)
        if len(title) == 0:
            return render_template('FQA.html', message='No question recommend!')
        ret = predict(title)
        print(ret)
        return render_template('FQA.html', message=ret, title=title)
    else:
        return render_template('FQA.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)