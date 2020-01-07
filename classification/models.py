import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DualEncoder(nn.Module):
    """
        简单的双编码器架构
    """
    def __init__(self,encoder1,encoder2,type="cosine"):
        super(DualEncoder,self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        if type == "CrossEntropy":
            #交叉熵损失的话增加两层线性神经网络输出，采用二分类交叉熵
            self.linear = nn.Sequential(
                nn.Linear(self.encoder1.hidden_size + self.encoder2.hidden_size,100),
                nn.ReLU(),
                nn.Linear(100,1)
            )

    def forward(self,x,x_mask,y,y_mask):
        x_rep = self.encoder1(x,x_mask)
        y_rep = self.encoder2(y,y_mask)
        return x_rep,y_rep

    def inference(self,x,x_mask,y,y_mask):
        x_rep,y_rep = self.forward(x,x_mask,y,y_mask)
        sim = F.cosine_similarity(x_rep,y_rep)

        return sim


class GRUEncoder(nn.Module):
    """
        简单的gru编码器
        avg_hidden:是否取平均，即是否用gru各层输出平均
    """
    def __init__(self,vocab_size,embed_size,hidden_size,dropout_p=0.1,avg_hidden=True,n_layers=1,bidirectional=True):
        super(GRUEncoder,self).__init__()
        self.hidden_size=hidden_size
        self.embed=nn.Embedding(vocab_size,embed_size)
        if bidirectional:
            hidden_size //= 2
        self.rnn = nn.GRU(embed_size,hidden_size,num_layers=n_layers,batch_first=True,bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout_p)
        self.bidirectional = bidirectional
        self.avg_hidden = avg_hidden

    def forward(self,x,mask):
        x_embed = self.dropout(self.embed(x))
        seq_len = mask.sum(1) #[batch_size],每个序列的长度
        # 处理padding的数据
        x_embed = nn.utils.rnn.pack_padded_sequence(
            x_embed,
            seq_len,
            batch_first=True,
            enforce_sorted=False
        )
        #output:[batch,seq,hidden*bidirectional]
        #hidden:[num_layers*bidirectional,batch,hidden]
        output,hidden = self.rnn(x_embed)
        output,seq_len = nn.utils.rnn.pad_packed_sequence(
            sequence=output,
            batch_first=True,
            padding_value=0.0,
            total_length=mask.shape[1]
        )

        #旧版本的pytorch
        # sorted_len,sorted_idx=seq_len.sort(0,descending=True)
        # x_sorted = x_embed[sorted_idx.long()]
        # x_embed = nn.utils.rnn.pack_padded_sequence(x_sorted,
        #                                             sorted_len.long().cpu().data.numpy(),
        #                                             batch_first=True)
        # output,hidden = self.rnn(x_embed)
        # output,_ = nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        # _,original_idx=sorted_idx.sort(0,descending=False)
        # output = output[original_idx.long()].contiguous()
        # hidden = hidden[:,original_idx.long()].contiguous()

        #是否取各层输出或直接取hidden，最后取seq维度上的平均
        if self.avg_hidden:
            hidden = torch.sum(output*mask.unsqueeze(2),1)/torch.sum(mask,1,keepdim=True)
        else:
            #取h_t
            if self.bidirectional:
                hidden = torch.cat((hidden[-2,:,:],hidden[-1,:,:]),dim=1)
            else:
                hidden = hidden[-1,:,:]
        hidden = self.dropout(hidden)

        #[batch,hidden_size]
        return hidden

    class TransformerEncoder(nn.Module):
        #TODO
        pass