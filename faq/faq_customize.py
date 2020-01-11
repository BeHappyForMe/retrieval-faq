import argparse
from collections import Counter
import code
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm,trange

from models_customize import GRUEncoder,DualEncoder,create_tokenizer,list2tensor,Tokenizer

"""
    自定义model，训练新问题与答案的相似度
"""

def train(args,model,tokenizer,optimizer,df,device):
    df = df[df['is_best']==1].sample(frac=1)
    df['neg_reply'] = df['reply'].sample(frac=1).tolist()
    df = df[df['neg_reply'] != df['reply']]

    model.train()
    for i in trange(0,df.shape[0],args.batch_size,desc='Iteration'):
        batch_df = df.iloc[i:i+args.batch_size]
        title = batch_df['best_title'].astype("str").tolist()
        reply = batch_df['reply'].astype("str").tolist()
        neg_reply = batch_df['neg_reply'].astype("str").tolist()
        batch_size = len(title)
        # 一半正例一半负例
        titles = title + title
        replies = reply + neg_reply
        x, x_mask = list2tensor(titles, tokenizer)
        y, y_mask = list2tensor(replies, tokenizer)
        target = x.new_ones(batch_size * 2).float()
        target[batch_size:] = 0 if args.loss_function == "CrossEntropy" else -1
        x = x.to(device)
        x_mask = x_mask.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        target = target.to(device)

        x_rep, y_rep = model(x, x_mask, y, y_mask)
        if args.loss_function == "cosine":
            loss_fn = nn.CosineEmbeddingLoss()
            loss = loss_fn(x_rep, y_rep, target)
            sim = F.cosine_similarity(x_rep, y_rep)
            sim[sim < 0] = -1
            sim[sim >= 0] = 1
            acc = torch.sum(sim == target).item() / target.shape[0]
        elif args.loss_function == "CrossEntropy":
            loss_fn = nn.BCEWithLogitsLoss()
            logits = model.linear(torch.cat([x_rep, y_rep], 1))
            loss = loss_fn(logits, target)
            sim = torch.sigmoid(logits)
            sim[sim < 0.5] = 0
            sim[sim >= 0.5] = 1
            acc = torch.sum(sim == target).item() / target.shape[0]
        elif args.loss_function == "Hinge":
            sim = F.cosine_similarity(x_rep, y_rep)
            sim1 = sim[:batch_size]
            sim2 = sim[batch_size:]
            loss = sim2 - sim1 + 0.5
            loss[loss < 0] = 0
            loss = torch.mean(loss)
            acc = torch.sum(sim1 > sim2).item() / batch_size

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if i % 100 == 0:
            print("iteration: {}, loss: {}, accuracy: {}".format(i, loss.item(), acc))

def evulate(args,model,tokenizer,df,device):
    df = df[df['is_best']==1].sample(frac=1)
    df['neg_reply'] = df['reply'].sample(frac=1).tolist()
    df = df[df['neg_reply'] != df['reply']]

    num_corrects, total_counts = 0, 0
    model.eval()
    for i in trange(0,df.shape[0],args.batch_size,desc='Iteration'):
        batch_df = df.iloc[i:i+args.batch_size]
        title = batch_df['best_title'].astype("str").tolist()
        reply = batch_df['reply'].astype("str").tolist()
        neg_reply = batch_df['neg_reply'].astype("str").tolist()
        batch_size = len(title)
        # 一半正例一半负例
        titles = title + title
        replies = reply + neg_reply
        x, x_mask = list2tensor(titles, tokenizer)
        y, y_mask = list2tensor(replies, tokenizer)
        target = x.new_ones(batch_size * 2).float()
        target[batch_size:] = 0 if args.loss_function == "CrossEntropy" else -1
        x = x.to(device)
        x_mask = x_mask.to(device)
        y = y.to(device)
        y_mask = y_mask.to(device)
        target = target.to(device)

        with torch.no_grad():
            x_rep, y_rep = model(x, x_mask, y, y_mask)
            if args.loss_function == "cosine":
                sim = F.cosine_similarity(x_rep, y_rep)
                sim[sim < 0] = -1
                sim[sim >= 0] = 1
                num_corrects += torch.sum(sim == target).item()
            elif args.loss_function == "CrossEntropy":
                logits = model.linear(torch.cat([x_rep, y_rep], 1))
                sim = torch.sigmoid(logits)
                sim[sim < 0.5] = 0
                sim[sim >= 0.5] = 1
                num_corrects += torch.sum(sim == target).item()
            elif args.loss_function == "Hinge":
                sim = F.cosine_similarity(x_rep, y_rep)
                sim1 = sim[:batch_size]
                sim2 = sim[batch_size:]
                num_corrects += torch.sum(sim1 > sim2).item()
            total_counts += batch_size

    print("accuracy:{}".format(num_corrects/total_counts))
    return num_corrects/total_counts


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="./data/train.csv", type=str, required=False,
						help="training file")
    parser.add_argument("--dev_file", default="./data/dev.csv", type=str, required=False,
						help="development file")
    parser.add_argument("--output_dir", default="./model_pkl/customize", type=str, required=False,
						help="output directory for tokenizers and models")
    parser.add_argument("--num_epochs", default=3, type=int, required=False,
						help="number of epochs for training")
    parser.add_argument("--vocab_size", default=50000, type=int, required=False,
						help="vocabulary size")
    parser.add_argument("--hidden_size", default=300, type=int, required=False,
						help="hidden size of GRU")
    parser.add_argument("--embed_size", default=300, type=int, required=False,
						help="word embedding size")
    parser.add_argument("--batch_size", default=8, type=int, required=False,
						help="batch size for train and eval")
    parser.add_argument("--loss_function", default="cosine", type=str, required=False,
						choices=["CrossEntropy", "cosine","Hinge"],
						help="which loss function to choose")
    args = parser.parse_args()

    #load data
    train_df = pd.read_csv(args.train_file)
    train_best_title = train_df.apply(lambda row:row['question'] if row['question'] is not None and
                                      len(str(row['question'])) > len(str(row['title']))
                                      else row['title'],axis=1)   #title与question处理
    train_df['best_title'] = train_best_title
    dev_df = pd.read_csv(args.dev_file)
    dev_best_title = dev_df.apply(lambda row:row['question'] if row['question'] is not None and
                                      len(str(row['question'])) > len(str(row['title']))
                                      else row['title'],axis=1)   #title与question处理
    dev_df['best_title'] = dev_best_title

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(os.path.join(args.output_dir,"customize_tokenizer.pickle")):
        texts = train_best_title.astype("str").tolist() + dev_best_title.astype("str").tolist()
        tokenizer = create_tokenizer(texts, args.vocab_size)
        with open(os.path.join(args.output_dir,"customize_tokenizer.pickle"),"wb") as fint:
            pickle.dump(tokenizer,fint)
    else:
        with open(os.path.join(args.output_dir,"customize_tokenizer.pickle"),"rb") as fout:
            tokenizer = pickle.load(fout)

    title_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
    reply_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
    model = DualEncoder(title_encoder,reply_encoder,args.loss_function)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_ac = 0.
    for epoch in trange(args.num_epochs,desc="Epoch"):
        print("strat epoch:{}".format(epoch))
        train(args,model,tokenizer,optimizer,train_df,device)
        acc = evulate(args,model,tokenizer,dev_df,device)
        if acc > best_acc:
            best_acc = acc
            print("saving best model")
            torch.save(model.state_dict(), os.path.join(args.output_dir, "faq_customize.pth"))

if __name__ == '__main__':
    main()

