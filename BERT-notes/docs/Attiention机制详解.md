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

![Attention model](https://pic4.zhimg.com/80/v2-163c0c3dda50d1fe7a4f7a64ba728d27_hd.jpg)

在该模型中，定义了一个条件概率：

![formula1](https://pic1.zhimg.com/80/v2-63ec36313044de9414c6ecc76814b6ec_hd.jpg)

其中，si是decoder中RNN在在i时刻的隐状态，如图4中所示，其计算公式为：

![formula2](https://pic3.zhimg.com/80/v2-de918a0ad5f38e2b2e199ae27b018b32_hd.jpg)

这里的背景向量ci的计算方式，与传统的Seq2Seq模型直接累加的计算方式不一样，这里的ci是一个权重化（Weighted）之后的值，其表达式如公式5所示：

![formula3](https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/公式1.PNG)

其中，i表示encoder端的第i个词，hj表示encoder端的第j和词的隐向量，aij表示encoder端的第j个词与decoder端的第i个词之间的权值，表示源端第j个词对目标端第i个词的影响程度，aij的计算公式如公式6所示：

![formula4](https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/公式2.PNG)

### self-Attention

Self Attention与传统的Attention机制非常的不同：传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，self Attention Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，self Attention可以不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系。
在Google提出的论文[《Attention is all you need》](https://arxiv.org/abs/1706.03762)中提出了一个新的完全基于注意力机制的机器翻译模型，其亮点在于有：1）不同于以往主流机器翻译使用基于RNN的seq2seq模型框架，该论文用attention机制代替了RNN搭建了整个模型框架。2）提出了多头注意力（Multi-headed attention）机制方法，在编码器和解码器中大量的使用了多头自注意力机制（Multi-headed self-attention）。3）在WMT2014语料中的英德和英法任务上取得了先进结果，并且训练速度比主流模型更快。通过对论文的阅读可以很好地去理解self-Attention的机理和强大之处。
在这篇论文中提到Attention函数的本质可以被描述为一个查询（query）到一系列（键key-值value）对的映射，如下图。

![attention_essence](https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_1.PNG)

#### 1. scaled dot-Product attention
scaled dot-Product attention就是我们常用的使用点积进行相似度计算的attention，只是多除了一个（为K的维度）起到调节作用，其作用在于：减小内积的大小。**防止当 dk 比较大时点乘结果太大导致有效梯度太大（或被忽略、裁剪）**。对于为什么除以的是根号d_k，在论文下方的注释里提到：假设两个 d_k 维向量每个分量都是一个相互独立的服从标准正态分布的随机变量，那么他们的点乘的方差就是 d_k，每一个分量除以 sqrt(d_k) 可以让点乘的方差变成 1。

![scaled dot-Product attention](https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_2.PNG)

#### 2. Multi-head attention
多头attention（Multi-head attention）结构如下图，Query，Key，Value首先进过一个线性变换，然后输入到放缩点积attention，注意这里要做h次，其实也就是所谓的多头，每一次算一个头。而且每次Q，K，V进行线性变换的参数W是不一样的。然后将h次的放缩点积attention结果进行拼接，再进行一次线性变换得到的值作为多头attention的结果。可以看到，google提出来的多头attention的不同之处在于进行了h次计算而不仅仅算一次，论文中说到这样的好处是可以允许模型在不同的表示子空间里学习到相关的信息，后面还会根据attention可视化来验证。

![Multi-head attention](https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_3.PNG)

那么在整个模型中，是如何使用attention的呢？如下图，首先在编码器到解码器的地方使用了多头attention进行连接，K，V，Q分别是编码器的层输出（这里K=V）和解码器中都头attention的输入。其实就和主流的机器翻译模型中的attention一样，利用解码器和编码器attention来进行翻译对齐。然后在编码器和解码器中都使用了多头自注意力self-attention来学习文本的表示。Self-attention即K=V=Q，例如输入一个句子，那么里面的每个词都要和该句子中的所有词进行attention计算。目的是学习句子内部的词依赖关系，捕获句子的内部结构。
那么在整个模型中，是如何使用attention的呢？如下图，首先在编码器到解码器的地方使用了多头attention进行连接，K，V，Q分别是编码器的层输出（这里K=V）和解码器中都头attention的输入。其实就和主流的机器翻译模型中的attention一样，利用解码器和编码器attention来进行翻译对齐。然后在编码器和解码器中都使用了多头自注意力self-attention来学习文本的表示。Self-attention即K=V=Q，例如输入一个句子，那么里面的每个词都要和该句子中的所有词进行attention计算。目的是学习句子内部的词依赖关系，捕获句子的内部结构。

![Attention in model](https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_4.PNG)

对于使用自注意力机制的原因，论文中提到主要从三个方面考虑（每一层的复杂度，是否可以并行，长距离依赖学习），并给出了和RNN，CNN计算复杂度的比较。可以看到，如果输入序列n小于表示维度d的话，每一层的时间复杂度self-attention是比较有优势的。当n比较大时，作者也给出了一种解决方案self-attention（restricted）即每个词不是和所有词计算attention，而是只与限制的r个词去计算attention。在并行方面，多头attention和CNN一样不依赖于前一时刻的计算，可以很好的并行，优于RNN。在长距离依赖上，由于self-attention是每个词和所有词都要计算attention，所以不管他们中间有多长距离，最大的路径长度也都只是1。可以捕获长距离依赖关系。

