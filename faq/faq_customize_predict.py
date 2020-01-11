import argparse
from collections import Counter
import code
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from models_customize import GRUEncoder,DualEncoder,create_tokenizer,list2tensor,Tokenizer

def prepare_replies(df,model,device,tokenizer,args):
    model.eval()
    vectors = []
    for i in range(0,df.shape[0],args.batch_size):
        batch_df = df.iloc[i:i+args.batch_size]
        reply = list(batch_df["reply"])
        y,y_mask = list2tensor(reply,tokenizer)
        y = y.to(device)
        y_mask = y_mask.to(device)

        y_rep = model.encoder2(y,y_mask)
        vectors.append(y_rep.data.cpu().numpy())

    replies = df["reply"].tolist()
    vectors = np.concatenate(vectors,0)
    return replies,vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="training file")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="output directory for tokenizers and models")
    parser.add_argument("--batch_size", default=8, type=int, required=False,
                        help="batch size for train and eval")
    parser.add_argument("--hidden_size", default=300, type=int, required=False,
                        help="hidden size of GRU")
    parser.add_argument("--embed_size", default=300, type=int, required=False,
                        help="word embedding size")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_file)
    train_best_title = train_df.apply(lambda row:row['question'] if row['question'] is not None and
                                      len(str(row['question'])) > len(str(row['title']))
                                      else row['title'],axis=1)   #title与question处理
    train_df['best_title'] = train_best_title
    tokenizer = pickle.load(open(os.path.join(args.output_dir,"customize_tokenizer.pickle"),"rb"))

    title_encode = GRUEncoder(tokenizer.vocab_size,args.embed_size,args.hidden_size)
    reply_encode = GRUEncoder(tokenizer.vocab_size,args.embed_size,args.hidden_size)
    model = DualEncoder(title_encode,reply_encode)
    model.load_state_dict(torch.load(os.path.join(args.output_dir,"faq_customize.pth")))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    candidate_file = os.path.join(args.output_dir, "reply_candidates.pickle")
    if not os.path.isfile(candidate_file):
        replies, vectors = prepare_replies(train_df, model, device, tokenizer, args)
        pickle.dump([replies, vectors], open(candidate_file, "wb"))
    else:
        replies, vectors = pickle.load(open(candidate_file, "rb"))

    while True:
        title = input("你的问题是？\n")
        if len(title.strip()) == 0:
            continue

        temp = []
        x,x_mask = list2tensor(temp.append(str(title)),tokenizer)
        x = x.to(device)
        x_mask = x_mask.to(device)

        x_rep = model.encoder1(x,x_mask).data.cpu().numpy()
        scores = cosine_similarity(x_rep,vectors)[0]
        index = np.argmax(scores)

        print("可能得答案：",replies[index])


if __name__ == '__main__':
    main()


