import argparse
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm,trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from models import GRUEncoder,DualEncoder
from utils import Tokenizer,create_tokenizer,list2tensor

def train(df,model,loss_fn,optimizer,device,tokenizer,args):
    model.train()
    df = df.sample(frac=1)
    for i in trange(0,df.shape[0],args.batch_size,desc="Iteration"):
        batch_df = df.iloc[i:i+args.batch_size]
        title = list(batch_df["best_title"].astype("str"))
        reply = list(batch_df["reply"].astype("str"))
        target = torch.from_numpy(batch_df["is_best"].to_numpy()).float().view(-1,1)
        # CosineEmbeddingLoss的话标签分为1，-1
        if args.loss_function == "cosine":
            target[target==0] = -1
        x,x_mask = list2tensor(title,tokenizer)
        y,y_mask = list2tensor(reply,tokenizer)

        x,x_mask,y,y_mask = x.to(device),x_mask.to(device),y.to(device),y_mask.to(device)
        target = target.to(device)

        x_rep,y_rep = model(x,x_mask,y,y_mask)
        if args.loss_function == "cosine":
            loss = loss_fn(x_rep,y_rep,target)
        elif args.loss_function == "CrossEntropy":
            logits = model.linear(torch.cat([x_rep,y_rep],1))
            loss = loss_fn(logits,target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()

        #评估acc
        if args.loss_function == "cosine":
            sim = F.cosine_similarity(x_rep,y_rep)
            sim[sim<0]=-1
            sim[sim>=0]=1
        elif args.loss_function == "CrossEntropy":
            sim = model.linear(torch.cat([x_rep,y_rep],1))
            sim = torch.sigmoid(sim)
            sim[sim<0.5]=0
            sim[sim>=0.5]=1
        sim = sim.view(-1)
        target = target.view(-1)
        acc = torch.sum(sim==target).item()/target.shape[0]
        if i%100 ==0:
            print("iteration: {}, loss: {}, accuracy: {}".format(i, loss.item(), acc))

def evaluate(df,model,loss_fn,device,tokenizer,args):
    model.eval()

    num_corrects,total_counts =0,0
    for i in trange(0, df.shape[0], args.batch_size, desc="Iteration"):
        batch_df = df.iloc[i:i + args.batch_size]
        title = list(batch_df["best_title"])
        reply = list(batch_df["reply"])
        target = torch.from_numpy(batch_df["is_best"].to_numpy()).float().view(-1, 1)
        # CosineEmbeddingLoss的话标签分为1，-1
        if args.loss_function == "cosine":
            target[target == 0] = -1
        x, x_mask = list2tensor(title, tokenizer)
        y, y_mask = list2tensor(reply, tokenizer)

        x, x_mask, y, y_mask = x.to(device), x_mask.to(device), y.to(device), y_mask.to(device)

        with torch.no_grad():
            x_rep, y_rep = model(x, x_mask, y, y_mask)
            if args.loss_function == "cosine":
                loss = loss_fn(x_rep, y_rep, target)
                sim = F.cosine_similarity(x_rep,y_rep)
                sim[sim < 0] = -1
                sim[sim >= 0] = 1
            elif args.loss_function == "CrossEntropy":
                logits = model.linear(torch.cat([x_rep, y_rep], 1))
                loss = loss_fn(logits,target)
                sim = torch.sigmoid(logits)
                sim[sim < 0.5] = 0
                sim[sim >= 0.5] = 1

        sim = sim.view(-1)
        target = target.view(-1)
        num_corrects = torch.sum(sim==target).item()
        total_counts += target.shape[0]

    print("accuracy:{}".format(num_corrects/total_counts))
    return num_corrects/total_counts



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="training file")
    parser.add_argument("--dev_file", default=None, type=str, required=True,
                        help="development file")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="output directory for tokenizers and models")
    parser.add_argument("--num_epochs", default=5, type=int, required=False,
                        help="number of epochs for training")
    parser.add_argument("--vocab_size", default=50000, type=int, required=False,
                        help="vocabulary size")
    parser.add_argument("--hidden_size", default=300, type=int, required=False,
                        help="hidden size of GRU")
    parser.add_argument("--embed_size", default=300, type=int, required=False,
                        help="word embedding size")
    parser.add_argument("--batch_size", default=16, type=int, required=False,
                        help="batch size for train and eval")
    parser.add_argument("--loss_function", default="CrossEntropy", type=str, required=False,
                        choices=["CrossEntropy", "cosine"],
                        help="which loss function to choose")
    args = parser.parse_args()

    #load dataset
    train_df = pd.read_csv(args.train_file)[["best_title","reply","is_best"]]
    print("the length of train:{}".format(len(train_df)))
    dev_df = pd.read_csv(args.dev_file)[["best_title","reply","is_best"]]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(os.path.join(args.output_dir,"tokenizer.pickle")):
        print("load tokenizer from file:{}".format(os.path.join(args.output_dir,"tokenizer.pickle")))
        with open(os.path.join(args.output_dir,"tokenizer.pickle"),"rb") as fint:
            tokenizer = pickle.load(fint)
    else:
        texts = list(train_df["best_title"].astype("str")) + list(train_df["reply"].astype("str"))
        tokenizer = create_tokenizer(texts, args.vocab_size)
        with open(os.path.join(args.output_dir, "tokenizer.pickle"), "wb") as fout:
            pickle.dump(tokenizer, fout)

    # TODO 使用简单的GRU结构
    title_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
    reply_encoder = GRUEncoder(tokenizer.vocab_size, args.embed_size, args.hidden_size)
    model = DualEncoder(title_encoder,reply_encoder,type=args.loss_function)
    print("the structure of model:{}".format(model))
    if args.loss_function == "CrossEntropy":
        loss_fn = nn.BCEWithLogitsLoss()
    elif args.loss_function == "cosine":
        loss_fn = nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_acc =0.
    for epoch in trange(args.num_epochs,desc="Epoch"):
        print("start epoch:{}".format(epoch))
        train(train_df,model,loss_fn,optimizer,device,tokenizer,args)
        acc = evaluate(dev_df, model, loss_fn, device, tokenizer, args)
        if acc>best_acc:
            best_acc=acc
            print("saving the best model")
            torch.save(model.state_dict(),os.path.join(args.output_dir,"class_model.pth"))


if __name__ == '__main__':
    main()