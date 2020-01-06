import torch
import pandas as pd
import numpy as np
import pkuseg
from collections import Counter

def split():
    """
        分割训练、测试集
    :return:
    """
    df = pd.read_csv("./data/baoxianzhidao_filter.csv")

    # 数据集中question为null情况
    best_title = df.apply(lambda row:row['question'] if row['question'] is not None and len(str(row['question'])) > len(str(row['title'])) else
                          row['title'],axis=1)
    df['best_title'] = best_title

    sample = np.random.choice(df.index,size=int(len(df)*0.9),replace=False)

    df.iloc[sample].to_csv("./data/train.csv",index=False)
    df.drop(sample).to_csv("./data/dev.csv",index=False)


class Tokenizer():
    """
        自定义tokenizer
    """
    def __init__(self,vocab):
        self.id2word = ["UNK"] + vocab
        self.word2id = {word:idx for idx,word in enumerate(self.id2word)}

    def text2id(self,text):
        return [self.word2id.get(w,0) for w in text]

    def id2text(self,ids):
        return "".join([self[id] for id in ids])

    @property
    def vocab_size(self):
        return len(self.id2word)

def create_tokenizer(texts,vocab_size):
    """
        从texts中构建tokenizer
    :param texts: list[str] of text
    :param vocab_size:
    :return:
    """
    allvocab = []
    word_counter = Counter()
    seg = pkuseg.pkuseg()
    for text in texts:
        text = seg.cut(text)
        for word in text:
            word_counter[word] +=1
    vocab = word_counter.most_common(vocab_size)
    vocab = [w[0] for w in vocab]
    return Tokenizer(vocab)

def list2tensor(sents,tokenizer):
    res = []
    mask = []
    seg = pkuseg.pkuseg()
    for sent in sents:
        res.append(tokenizer.text2id(seg.cut(sent)))

    max_len = max([len(sen) for sen in res])
    for i in range(len(res)):
        _mask = np.zeros((1,max_len))
        _mask[:,:len(res[i])] = 1
        #这里先构建mask，不然res[i]改变后原长度找不到了
        res[i] = np.expand_dims(np.array(res[i] + [0]*(max_len-len(res[i]))),0)
        mask.append(_mask)
    res = np.concatenate(res,axis=0)
    mask = np.concatenate(mask,axis=0)
    res = torch.from_numpy(res).long()
    mask = torch.from_numpy(mask).long()
    return res,mask
