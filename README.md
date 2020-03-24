# retrieval-faq

## 投资知道 基于BERT的中文最佳答案推荐

本文基于BERT中文预训练模型，使用huggingface transformers开源库实现中文关于投资知道的
问题最佳答案推荐系统的模型实现。问题适用场景：投资问答/论坛等，最佳答案推荐。
本文实现了2个baseline，分别为ELMo、BERT模型，都是较强的基准模型。

主模型基于BERT预训练模型进行fine-tuning。模型的关键是判断句子之间的相似度，考虑训练一个同义句模型，使用Hinge Loss损失函数。第一想法是构建同义句pair与非同义句pair，同义句pair的分数一定要比非同义句pair高，**max( - (同义句的分数 - 非同义句的分数 - margin), 0 )**，训练模型后发现效果非常差，比不上baseline模型，且没有很好的评估方式。cosine similarity的方式应该更好，考虑使用同义句之间的cosine similarity分数与非同义句的cosine similarity分数之间做Hinge Loss。



### 数据集

数据来源于<a href="https://github.com/SophonPlus/ChineseNlpCorpus" target="_blank">ChineseNlpCorpus</a>，58.8W条投资保险知道。
数据集预处理方式在data文件夹下preprocess_data_touzi.ipynb。同义句获取使用回译的方式，借助
有道云api将原问题翻译为英文后再翻译回中文，验证数据为自己手动整理查找的大概50条。

数据集下载链接：<a href="https://pan.baidu.com/s/1l3ttWMTdrp1sFBjjS-eVzw" target="_blank">百度网盘</a>，提取码：g1ug。


### 基线模型

#### ELMo：
ELMo使用Bi_LSTM拼接的方式来获取双向语义信息，下面是具体使用方法：

* 安装PyTorch

* 安装allennlp，ELMo的具体实现
pip install allennlp

* 下载 ELMoForManyLangs，提供中文支持

    git clone https://github.com/HIT-SCIR/ELMoForManyLangs.git
    
    cd ELMoForManyLangs
    
    python setup.py install (注意ELMoForManyLangs要求python >= 3.6)

* 下载简体中文版预训练ELMo模型

    百度云链接 <a href="https://pan.baidu.com/s/1RNKnj6hgL-2orQ7f38CauA?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&trace">中文ELMo</a>
    。注意从百度云下载并untar模型之后，需要把config.json当中的config_path修改成elmoformanylangs当中提供的config包里的json文件路径。

* 安装北大分词工具：pip install pkuseg 。（这步不是必须，也可以用其他中文分词工具包或者按字分割

ELMo模型最终MRR指标为：0.18630804064680587


####BERT

直接使用BERT来对问题进行编码，采用CLS token或者在sequence上求平均来作为句子的表示，使用的为huggingface 的transformers，
模型训练及评估代码在bert_baseline包下bert_base.py中。MRR指标为0.1910977126696094。


#### BERT微调

使用回译获取的句子与原问题看做同义句，在原数据集中随机采样数据后取tf-idf最高的与原问题组成
相似度较高的非同义句（增加模型识别难道），对同义句与非同义句分别求cosine similarity，Hinge loss
为目标函数，基于BERT预训练模型进行fine-tuning，模型训练及验证代码在bert_synonymous.py中。模型最终MRR：0.19247115729927698

结果并不理想，可能的原因分析：

* round trip translation的翻译效果不佳，很多句子与原句的语义有较大差距。
* 数据量太小

<a href="https://github.com/BeHappyForMe/retrieval-faq/blob/master/image/faq_0.png">结果展示</a>