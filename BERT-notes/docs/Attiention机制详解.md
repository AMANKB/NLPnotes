## Attention机制

Attention Mechanism与人类对外界事物的观察机制很类似，当人类观察外界事物的时候，一般不会把事物当成一个整体去看，往往倾向于根据需要选择性的去获取被观察
事物的某些重要部分，比如我们看到一个人时，往往先Attention到这个人的脸，然后再把不同区域的信息组合起来，形成一个对被观察事物的整体印象。因此Attention 
Mechanism可以帮助模型对输入的X每个部分赋予不同的权重，抽取出更加关键及重要的信息，使模型做出更加准确的判断，同时不会对模型的计算和存储
带来更大的开销.

### 为什么需要机制Attention
提出Attention机制是因为原有的seq2seq模型所存在的一些问题。下图是一个经典的seq2seq模型，源自论文《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation
》<https://arxiv.org/abs/1406.1078>

![seq2seq2](/images/seq2seq.png)
图1 经典的seq2seq2模型

其中，Encoder把一个变成的输入序列x1，x2，x3....xt编码成一个固定长度隐向量（背景向量，或上下文向量context）c，c有两个作用：
1. 做为初始向量初始化Decoder的模型，做为decoder模型预测y1的初始向量。
2. 做为背景向量，指导y序列中每一个step的y的产出。Decoder主要基于背景向量c和上一步的输出yt-1解码得到该时刻t的输出yt，直到碰到结束标志（<EOS>）为止。

《Sequence to Sequence Learning with Neural Networks》介绍了一种基于RNN的Seq2Seq模型，基于一个Encoder和一个Decoder来构建基于神经网络的End-to-End的机器翻译模型，其中，Encoder把输入X编码成一个固定长度的隐向量Z，Decoder基于隐向量Z解码出目标输出Y。
这是一个非常经典的序列到序列的模型，但是却存在两个明显的问题：


