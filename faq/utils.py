from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def tfidf_similarity(cv, s, texts):
    s, sents = ' '.join(list(s)), [' '.join(list(text)) for text in texts]
    corpus = [s] + sents
    vectors = cv.fit_transform(corpus).toarray()
    scores = cosine_similarity(vectors[0].reshape(1, -1), vectors[1:])
    # print(scores)
    sorted_indices = scores.argsort()[0, ::-1]
    return texts[sorted_indices[0]]


def merge():
    translated = pd.read_csv('./data/touzi_synonymous.csv', sep='\t', header=0)
    print("translated length:{}".format(len(translated)))
    pd_all = pd.read_csv('./data/right_samples.csv', sep='\t')
    print("right_samples length:{}".format(len(pd_all)))
    merged = pd_all.merge(translated, left_on='best_title', right_on='best_title')
    print("merged length: {}".format(len(merged)))
    merged = merged[['best_title', 'translated', 'reply', 'is_best']]
    merged.drop_duplicates(inplace=True)
    print("drop_duplicates merged length: {}".format(len(merged)))
    merged.to_csv('./data/touzi_preprocessed_synonymous.csv', index=False, sep='\t')