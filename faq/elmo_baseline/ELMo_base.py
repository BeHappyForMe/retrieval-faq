import argparse
import os
import pickle
import logging
import logging.handlers
from sklearn.metrics.pairwise import cosine_similarity
import code
from tqdm import trange, tqdm

import numpy as np
import pandas as pd
from elmoformanylangs import Embedder
import pkuseg
import sys

sys.path.append("..")
from metric import mean_reciprocal_rank, mean_average_precision

e = Embedder('D:\\NLP\\my-wholes-models\\zhs.model')
seg = pkuseg.pkuseg()

logger = logging.getLogger(__name__)
handler2 = logging.FileHandler(filename="ELMo.log")
logger.setLevel(logging.DEBUG)
handler2.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
handler2.setFormatter(formatter)
logger.addHandler(handler2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default='../data/right_samples.csv', type=str, required=False,
                        help="training file")
    parser.add_argument("--evaluate_file", default='../data/eval_touzi.xlsx', type=str, required=False,
                        help="training file")
    parser.add_argument("--do_evaluate", action="store_true", help="Whether to run evaluate.")
    parser.add_argument("--do_predict", default=True, action="store_true", help="Whether to run predict.")

    parser.add_argument("--batch_size", default=16, type=int, required=False,
                        help="batch size for train and eval")
    args = parser.parse_args()

    if not os.path.exists("embeddings.pkl"):
        train_df = pd.read_csv(args.train_file, sep='\t')
        candidate_title = train_df['best_title'].tolist()
        candidate_reply = train_df["reply"].tolist()

        titels = []
        for title in tqdm(candidate_title, desc='对原问题进行分词ing'):
            titels.append(seg.cut(title))
        embeddings = []
        for i in trange(0, len(titels), 16, desc='获取ELMo的句子表示'):
            mini_embeddings = e.sents2elmo(titels[i:min(len(titels), i + 16)])
            for mini_embedding in mini_embeddings:
                # 获取句子向量，对词取平均
                embeddings.append(np.mean(mini_embedding, axis=0))
            if i == 0:
                print(len(embeddings))
                print(embeddings[0].shape)
        print("原始问题句子向量表示获取完毕，保存ing")
        with open("embeddings.pkl", 'wb') as fout:
            pickle.dump([candidate_title, candidate_reply, embeddings], fout)
    else:
        with open("embeddings.pkl", 'rb') as fint:
            candidate_title, candidate_reply, embeddings = pickle.load(fint)

    if args.do_evaluate:
        evulate_df = pd.read_excel(args.evaluate_file, '投资知道')
        # code.interact(local=locals())
        evulate_df = evulate_df[['问题', '匹配问题']]
        evulate_df = evulate_df[evulate_df['问题'].notna()]
        evulate_df = evulate_df[evulate_df['匹配问题'].notna()]

        questions = evulate_df['问题'].tolist()
        matched_questions = evulate_df['匹配问题'].tolist()
        matched_questions_indexs = []
        for k, q in enumerate(matched_questions):
            flag = False
            for i, _q in enumerate(candidate_title):
                if q == _q:
                    matched_questions_indexs.append([i])
                    flag = True
                    break
            if not flag:
                matched_questions_indexs.append([-1])

        matched_questions_indexs = np.asarray(matched_questions_indexs)
        # print("size of matched_questions_index:{}".format(matched_questions_indexs.shape))

        questions = [seg.cut(question.strip()) for question in questions]
        question_embedding = [np.mean(emb, 0) for emb in e.sents2elmo(questions)]
        scores = cosine_similarity(question_embedding, embeddings)
        # print("scores shape : {}".format(scores.shape))
        sorted_indices = scores.argsort()[:, ::-1]
        # code.interact(local=locals())

        mmr = mean_reciprocal_rank(sorted_indices == matched_questions_indexs)
        map = mean_average_precision(sorted_indices == matched_questions_indexs)
        logger.info("mean reciprocal rank: {}".format(mmr))
        logger.info("mean average precision: {}".format(map))

    if args.do_predict:
        while True:
            title = input("你的问题是？\n")
            if len(str(title).strip()) == 0:
                continue
            title = [seg.cut(str(title).strip())]
            title_embedding = np.mean(e.sents2elmo(title)[0], 0).reshape(1, -1)
            scores = cosine_similarity(title_embedding, embeddings)[0]
            top5_indices = scores.argsort()[-5:][::-1]
            for index in top5_indices:
                print("可能的答案，参考问题：" + candidate_title[index] + "\t答案：" + candidate_reply[index] + "\t得分：" + str(
                    scores[index]))


if __name__ == '__main__':
    main()
