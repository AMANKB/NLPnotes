## Attention机制

Attention Mechanism与人类对外界事物的观察机制很类似，当人类观察外界事物的时候，一般不会把事物当成一个整体去看，往往倾向于根据需要选择性的去获取被观察
事物的某些重要部分，比如我们看到一个人时，往往先Attention到这个人的脸，然后再把不同区域的信息组合起来，形成一个对被观察事物的整体印象。**因此Attention 
Mechanism可以帮助模型对输入的X每个部分赋予不同的权重，抽取出更加关键及重要的信息，使模型做出更加准确的判断，同时不会对模型的计算和存储带来更大的开销.**

### 为什么需要机制Attention

提出Attention机制是因为原有的seq2seq模型所存在的一些问题。下图是一个经典的seq2seq模型，源自论文[《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》](https://arxiv.org/abs/1406.1078)

![seq2seq2](https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/seq2seq.PNG)

其中，Encoder把一个变成的输入序列x1，x2，x3....xt编码成一个固定长度隐向量（背景向量，或上下文向量context）c，c有两个作用：
* 做为初始向量初始化Decoder的模型，做为decoder模型预测y1的初始向量。
* 做为背景向量，指导y序列中每一个step的y的产出。Decoder主要基于背景向量c和上一步的输出yt-1解码得到该时刻t的输出yt，直到碰到结束标志（<EOS>）为止。
《Sequence to Sequence Learning with Neural Networks》介绍了一种基于RNN的Seq2Seq模型，基于一个Encoder和一个Decoder来构建基于神经网络的End-to-End的机器翻译模型，其中，Encoder把输入X编码成一个固定长度的隐向量Z，Decoder基于隐向量Z解码出目标输出Y。这是一个非常经典的序列到序列的模型，但是却存在两个明显的问题：
1. 把输入X的所有信息有压缩到一个固定长度的隐向量Z，忽略了输入输入X的长度，当输入句子长度很长，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。
2. 把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的，比如，在机器翻译里，输入的句子与输出句子之间，往往是输入一个或几个词对应于输出的一个或几个词。因此，对输入的每个词赋予相同权重，这样做没有区分度，往往是模型性能下降。

### Attention的原理

![Attention模块](https://pic4.zhimg.com/80/v2-163c0c3dda50d1fe7a4f7a64ba728d27_hd.jpg)

在该模型中，定义了一个条件概率：

(https://pic1.zhimg.com/80/v2-63ec36313044de9414c6ecc76814b6ec_hd.jpg)

其中，si是decoder中RNN在在i时刻的隐状态，如图4中所示，其计算公式为：

(https://pic3.zhimg.com/80/v2-de918a0ad5f38e2b2e199ae27b018b32_hd.jpg)

这里的背景向量ci的计算方式，与传统的Seq2Seq模型直接累加的计算方式不一样，这里的ci是一个权重化（Weighted）之后的值，其表达式如公式5所示：

(https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/公式1.PNG)

其中，i表示encoder端的第i个词，hj表示encoder端的第j和词的隐向量，aij表示encoder端的第j个词与decoder端的第i个词之间的权值，表示源端第j个词对目标端第i个词的影响程度，aij的计算公式如公式6所示：

(https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/公式2.PNG)


