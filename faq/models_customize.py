import torch
import torch.nn as nn
import torch.nn.functional as F
import pkuseg
from collections import Counter
import numpy as np


class GRUEncoder(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,dropout_p=0.1,avg_hidden=True,n_layers=2,bidirectional=True):
        super(GRUEncoder,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.hidden_size = hidden_size
        if bidirectional:
            hidden_size //=2
        self.gru = nn.GRU(embed_size,hidden_size,num_layers=n_layers,batch_first=True,
                          dropout=dropout_p,bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)
        self.bidirectional = bidirectional
        self.avg_hidden = avg_hidden

    def forward(self,x,x_mask):
        #[batch,seq,embed]
        x_embeded = self.dropout(self.embed(x))
        #一维,每个句子的seq长度
        seq_len = x_mask.sum(1)
        x_embeded = nn.utils.rnn.pack_padded_sequence(x_embeded,seq_len,
                                                      batch_first=True,enforce_sorted=False)

        #output: [batch,seq,hidden*bidirection] 最后一层输出
        #hidden: [num_lays*bidirection,batch,hidden]  每一层的最后一个seq输出
        output,hidden = self.gru(x_embeded)
        output,_ = nn.utils.rnn.pad_packed_sequence(output,batch_first=True,
                                                    padding_value=0.0,total_length=x_mask.shape[1])

        if self.avg_hidden:
            # 取最后一层输出，压扁平均seq维度
            hidden = torch.sum(output * x_mask.unsqueeze(2),dim=1) / torch.sum(x_mask,1,keepdim=True)
        else:
            if self.bidirectional:
                hidden = torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1)
            else:
                hidden = hidden[-1,:,:]

        #batch,hidden
        return self.dropout(hidden)

class TransformerEncoder(nn.Module):
    #TODO
    pass

class DualEncoder(nn.Module):
    """双编码器架构"""
    def __init__(self,encoder1,encoder2,type="cosine"):
        super(DualEncoder,self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        if type == 'CrossEntropy':
            self.linear = nn.Sequential(
                nn.Linear(self.encoder1.hidden_size+self.encoder2.hidden_size,100),
                nn.ReLU(),
                nn.Linear(100,1)
            )

    def forward(self,x,x_mask,y,y_mask):
        x_rep = self.encoder1(x,x_mask)
        y_rep = self.encoder2(y,y_mask)
        return x_rep,y_rep

    def interence(self,x,x_mask,y,y_mask):
        x_rep,y_rep = self.forward(x,x_mask,y,y_mask)
        sim = F.cosine_similarity(x_rep,y_rep)
        return sim


class Tokenizer():
    def __init__(self,vocab):
        self.id2word = ["UNK"] + vocab
        self.word2id = {word:idx for idx,word in enumerate(self.id2word)}

    def text2ids(self,text):
        return [self.word2id.get(word,0) for word in text]

    def id2text(self,ids):
        return "".join([self.id2word[id] for id in ids])

    @property
    def vocab_size(self):
        return len(self.id2word)


def create_tokenizer(texts,vocab_size):
    """

    :param texts: list of str
    :param vocab_size:
    :return:
    """
    seg = pkuseg.pkuseg()
    vocab_count = Counter()
    for text in texts:
        text = seg.cut(text)
        for word in text:
            vocab_count[word] += 1
    vocab = vocab_count.most_common(vocab_size)
    vocab = [w[0] for w in vocab]
    return Tokenizer(vocab)


def list2tensor(sents,tokenizer):

    seg = pkuseg.pkuseg()
    reps = []
    mask = []
    for sent in sents:
        reps.append(tokenizer.text2ids(seg.cut(sent)))

    max_len = max([len(r) for r in reps])
    for idx in range(len(reps)):
        _mask = np.zeros((1,max_len))
        _mask[:,:len(reps[idx])] = 1
        #构建mask
        mask.append(_mask)
        # pad 输入
        reps[idx] = np.expand_dims(np.array(reps[idx] + [0]*(max_len-len(reps[idx]))),0)

    reps = np.concatenate(reps,axis=0)
    mask = np.concatenate(mask,axis=0)

    reps = torch.from_numpy(reps).long()
    mask = torch.from_numpy(mask).float()
    return reps,mask

import pandas as pd

def split():
    """
        数据预处理
        分割训练、测试集
    :return:
    """
    df = pd.read_csv("./data/baoxianzhidao_filter.csv")

    sample = np.random.choice(df.index,size=int(len(df)*0.9),replace=False)

    df.iloc[sample].to_csv("./data/train.csv",index=False)
    df.drop(sample).to_csv("./data/dev.csv",index=False)

# split()

