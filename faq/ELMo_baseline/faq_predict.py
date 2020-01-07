import argparse
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import pkuseg
from elmoformanylangs import Embedder

e = Embedder('/Users/zhoup/develop/NLPSpace/my-pre-models/zhs.model')
seg = pkuseg.pkuseg()

# sents = ["我是一个很容易满足的人","小小的事都可以满足我"]
# sents = [seg.cut(sent) for sent in sents]
# embeddings = e.sents2elmo(sents)
# embeddings = [np.mean(embed,0) for embed in embeddings]
# print(embeddings[0].shape)
# print(embeddings[1].shape)
# score = cosine_similarity(embeddings[0].reshape(1,-1),embeddings[1].reshape(1,-1))
# print(score)
# print(score.shape)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="training file")
    parser.add_argument("--batch_size", default=16, type=int, required=False,
                        help="batch size for train and eval")
    args = parser.parse_args()

    #load data
    if not os.path.exists("embeddings.pkl"):
        train_df = pd.read_csv(args.train_file)
        candidates = train_df[train_df["is_best"] == 1]
        best_title = candidates.apply(
            lambda row: row['question'] if row['question'] is not None and len(str(row['question'])) > len(
                str(row['title'])) else row['title'], axis=1)
        candidate_title = best_title.tolist()
        candidate_reply = candidates["reply"].tolist()

        titels = [seg.cut(title) for title in candidate_title]
        # 一个list of narray
        embeddings = e.sents2elmo(titels)
        candidate_embeddings = [np.mean(embedding,0) for embedding in embeddings]
        with open("embeddings.pkl","wb") as fout:
            pickle.dump([candidate_title,candidate_reply,candidate_embeddings],fout)

    else:
        with open("embeddings.pkl","rb") as fint:
            candidate_title,candidate_reply,candidate_embeddings = pickle.load(fint)

    while True:
        title = input("你的问题是？\n")
        if len(str(title).strip()) == 0:
            continue
        title = [seg.cut(str(title).strip())]
        title_embedding = np.mean(e.sents2elmo(title)[0],0).reshape(1,-1)
        scores = cosine_similarity(title_embedding,candidate_embeddings)[0]
        print(scores.shape)
        top5_indices = scores.argsort()[-5:][::-1]
        for index in top5_indices:
            print("可能的答案，参考问题：" + candidate_title[index] + "\t答案：" + candidate_reply[index] + "\t得分：" + str(scores[index]))



if __name__ == '__main__':
    main()

