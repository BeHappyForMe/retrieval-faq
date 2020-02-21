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
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from transformers import glue_convert_examples_to_features as convert_examples_to_features

import pandas as pd
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm

logger = logging.getLogger(__name__)


def convert_one_example_to_features(examples, tokenizer, max_length=256, pad_token=0, pad_token_segment_id=0,
                                    mask_padding_with_zero=True):
    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=[0,1],
        max_length=max_length,
        output_mode='classification',
        pad_on_left=False,  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )

    return features


class FaqProcessor(DataProcessor):

    def _create_one_example(self, title, reply):
        examples = [InputExample(guid=0,text_a=title,text_b=reply,label=1)]
        return examples

    def prepare_replies(self, path):
        df = pd.read_csv(path)
        df = df.fillna(0)
        replies = [str(t) for t in df['reply'].tolist()]
        return replies


def getSimiTitleAnswers(new_title, all_title, total_df):
    def tfidf_similarity(s1, s2):
        def add_space(s):
            return ' '.join(list(s))

        # 将字中间加入空格
        s1, s2 = add_space(s1), add_space(s2)
        # 转化为TF矩阵
        cv = TfidfVectorizer(tokenizer=lambda s: s.split())
        corpus = [s1, s2]
        vectors = cv.fit_transform(corpus).toarray()
        # 计算TF系数
        return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

    sim_reply = []
    for tt in all_title:
        if tfidf_similarity(tt, new_title) > 0.5:
            print('sim title:', tt + '\n')
            tt_replies = total_df[total_df.best_title == tt].reply.tolist()

            if len(tt_replies) > 1:
                for i in tt_replies:
                    if i not in sim_reply:
                        sim_reply.append(i)
            else:
                if tt_replies not in sim_reply:
                    sim_reply.extend(tt_replies)
    return sim_reply

processor = FaqProcessor()


def predict(title, replies, tokenizer, model, device):
    examples = processor._create_one_example(title, replies)
    features = convert_one_example_to_features(examples, tokenizer)
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    result = False
    with torch.no_grad():
        inputs = {'input_ids': all_input_ids.to(device),
                  'attention_mask': all_attention_mask.to(device),
                  'token_type_ids': all_token_type_ids.to(device)}
        outputs = model(**inputs)
        logits = outputs[0]
        result = F.softmax(logits, dim=1)[0].argmax() == 1

    if(result):
        return "匹配正确"
    else:
        return "检测不匹配"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('models/touzi/checkpoint-330000')
model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained('models/touzi/checkpoint-330000')


app = Flask(__name__)

app.secret_key = 'F12Zr47j\3yX R~X@H!jLwf/T'


@app.route("/")
def hello_world():
    return render_template('FQA.html')


@app.route('/fqa', methods=['GET'])
def fqa():
    while request.args.get('title'):
        title = str(request.args.get('title'))
        reply = str(request.args.get('reply'))
        print(title)
        print(reply)
        if len(reply) == 0:
            return render_template('FQA.html', message='No answers recommend!')
        ret = predict(title, reply, tokenizer, model, device)
        print(ret)
        return render_template('FQA.html', message=ret, title=title, reply=reply)
    else:
        return render_template('FQA.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)