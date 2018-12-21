## Attention机制

Attention Mechanism与人类对外界事物的观察机制很类似，当人类观察外界事物的时候，一般不会把事物当成一个整体去看，往往倾向于根据需要选择性的去获取被观察
事物的某些重要部分，比如我们看到一个人时，往往先Attention到这个人的脸，然后再把不同区域的信息组合起来，形成一个对被观察事物的整体印象。**因此Attention 
Mechanism可以帮助模型对输入的X每个部分赋予不同的权重，抽取出更加关键及重要的信息，使模型做出更加准确的判断，同时不会对模型的计算和存储带来更大的开销.**

### seq2seq模型

提出Attention机制是因为原有的seq2seq模型所存在的一些问题。所以理解seq2seq的优缺点是必要的。下图是一个经典的seq2seq模型，源自论文[《Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation》](https://arxiv.org/abs/1406.1078)

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/seq2seq.PNG"/>
</div>

其中，Encoder把一个变成的输入序列x1，x2，x3....xt编码成一个固定长度隐向量（背景向量，或上下文向量context）c，c有两个作用：
* 做为初始向量初始化Decoder的模型，做为decoder模型预测y1的初始向量。
* 做为背景向量，指导y序列中每一个step的y的产出。Decoder主要基于背景向量c和上一步的输出yt-1解码得到该时刻t的输出yt，直到碰到结束标志（<EOS>）为止。
《Sequence to Sequence Learning with Neural Networks》介绍了一种基于RNN的Seq2Seq模型，基于一个Encoder和一个Decoder来构建基于神经网络的End-to-End的机器翻译模型，其中，Encoder把输入X编码成一个固定长度的隐向量Z，Decoder基于隐向量Z解码出目标输出Y。

整个的流程用下图来看更容易理解：

<div align=center>
<src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/seq2seq_3.png"/>
</div>

1. 所有输出端，都以一个通用的<start>标记开头，以<end>标记结尾，这两个标记也视为一个词/字；
2. 将<start>输入decoder，然后得到隐藏层向量，将这个向量与encoder的输出混合，然后送入一个分类器，分类器的结果应当输出$\boldsymbol{P}$；
3. 将$\boldsymbol{P}$输入decoder，得到新的隐藏层向量，再次与encoder的输出混合，送入分类器，分类器应输出$\boldsymbol{Q}$
4. 依此递归，直到分类器的结果输出<end>。

### 为什么需要Attention？
这是一个非常经典的序列到序列的模型，但是却存在两个明显的问题：
1. 把输入X的所有信息有压缩到一个固定长度的隐向量Z，忽略了输入输入X的长度，当输入句子长度很长，特别是比训练集中最初的句子长度还长时，模型的性能急剧下降。
2. 把输入X编码成一个固定的长度，对于句子中每个词都赋予相同的权重，这样做是不合理的，比如，在机器翻译里，输入的句子与输出句子之间，往往是输入一个或几个词对应于输出的一个或几个词。因此，对输入的每个词赋予相同权重，这样做没有区分度，往往是模型性能下降。

### Attention的原理

<div align=center><img src="https://pic4.zhimg.com/80/v2-163c0c3dda50d1fe7a4f7a64ba728d27_hd.jpg"></div>

在该模型中，定义了一个条件概率：

<div align=center>
<img src="https://pic1.zhimg.com/80/v2-63ec36313044de9414c6ecc76814b6ec_hd.jpg"/>
</div>

其中，si是decoder中RNN在在i时刻的隐状态，如图4中所示，其计算公式为：

<div align=center>
<src="https://pic3.zhimg.com/80/v2-de918a0ad5f38e2b2e199ae27b018b32_hd.jpg"/>
</div>

这里的背景向量ci的计算方式，与传统的Seq2Seq模型直接累加的计算方式不一样，这里的ci是一个权重化（Weighted）之后的值，其表达式如公式5所示：

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/公式1.PNG"/>
</div>
其中，i表示encoder端的第i个词，hj表示encoder端的第j和词的隐向量，aij表示encoder端的第j个词与decoder端的第i个词之间的权值，表示源端第j个词对目标端第i个词的影响程度，aij的计算公式如公式6所示：

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/公式2.PNG"/>
</div>

### self-Attention

Self Attention与传统的Attention机制的不同在于：传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系，也就是**在序列的内部做Attention，寻找序列内部的联系**；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，self Attention Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，self Attention可以不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系。
在Google提出的论文[《Attention is all you need》](https://arxiv.org/abs/1706.03762)中提出了一个新的完全基于注意力机制的机器翻译模型，其亮点在于有：1）不同于以往主流机器翻译使用基于RNN的seq2seq模型框架，该论文用attention机制代替了RNN搭建了整个模型框架。2）提出了多头注意力（Multi-headed attention）机制方法，在编码器和解码器中大量的使用了多头自注意力机制（Multi-headed self-attention）。3）在WMT2014语料中的英德和英法任务上取得了先进结果，并且训练速度比主流模型更快。通过对论文的阅读可以很好地去理解self-Attention的机理和强大之处。
在这篇论文中提到Attention函数的本质可以被描述为一个查询（query）到一系列（键key-值value）对的映射，如下图。

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/attention_1.PNG"/>
</div>

在计算attention时主要分为三步，第一步是将query和每个key进行相似度计算得到权重，常用的相似度函数有点积，拼接，感知机等；然后第二步一般是使用一个softmax函数对这些权重进行归一化；最后将权重和相应的键值value进行加权求和得到最后的attention。目前在NLP研究中，key和value常常都是同一个，即key=value。

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_2.PNG"/>
</div>

该论文模型的整体结构如下图，还是由编码器和解码器组成，在编码器的一个网络块中，由一个多头attention子层和一个前馈神经网络子层组成，整个编码器栈式搭建了N个块。类似于编码器，只是解码器的一个网络块中多了一个多头attention层。为了更好的优化深度网络，整个网络使用了残差连接和对层进行了规范化（Add&Norm）。

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_6.png"/>
</div>

#### 1. scaled dot-Product attention
scaled dot-Product attention就是我们常用的使用点积进行相似度计算的attention，只是多除了一个（为K的维度）起到调节作用，其作用在于：减小内积的大小。**防止当 dk 比较大时点乘结果太大导致有效梯度太大（或被忽略、裁剪）**。对于为什么除以的是根号$d_k$，在论文下方的注释里提到：假设两个 d_k 维向量每个分量都是一个相互独立的服从标准正态分布的随机变量，那么他们的点乘的方差就是$d_k$个分量除以$sqrt {{d_k}}$以让点乘的方差变成 1。

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/attention_3.png"/>
</div>

对上面式子的理解：$\boldsymbol{Q}\in\mathbb{R}^{n\times d_k}, \boldsymbol{K}\in\mathbb{R}^{m\times d_k}, \boldsymbol{V}\in\mathbb{R}^{m\times d_v}$.如果忽略激活函数softmax的话，那么事实上它就是三个$\times d_k,d_k\times m, m\times d_v$矩阵相乘，
最后的结果是一个$n\times d_v$的矩阵。所以这里的操作可以被看作是**一个Attention层，将$n\times d_k$的序列$\boldsymbol{Q}$编码成了一个新的$n\times d_v$的序列。**
通过单个向量来看的话，
$$Attention(\boldsymbol{q}_t,\boldsymbol{K},\boldsymbol{V}) = \sum_{s=1}^m \frac{1}{Z}\exp\left(\frac{\langle\boldsymbol{q}_t, \boldsymbol{k}_s\rangle}{\sqrt{d_k}}\right)\boldsymbol{v}_s$$
其中Z是归一化因子。事实上q,k,v分别是query,key,value的简写，K,V是一一对应的，它们就像是key-value的关系，那么上式的意思就是通过qt这个query，通过与各个ks内积的并softmax的方式，来得到$q_t$与各个$v_s$的相似度，然后加权求和，得到一个$d_v$维的向量。
#### 2. Multi-head attention
多头attention（Multi-head attention）结构如下图，Query，Key，Value首先进过一个线性变换，然后输入到放缩点积attention，注意这里要做h次，其实也就是所谓的多头，每一次算一个头。而且每次Q，K，V进行线性变换的参数W是不一样的(**参数不共享**)。然后**将h次的放缩点积attention结果进行拼接**，再进行一次线性变换得到的值作为多头attention的结果。可以看到，google提出来的多头attention的不同之处在于进行了h次计算而不仅仅算一次，论文中说到这样的好处是可以允许模型在不同的表示子空间里学习到相关的信息，后面还会根据attention可视化来验证。

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_4.png"/>
</div>

这里的$\boldsymbol{W}_i^Q\in\mathbb{R}^{d_k\times \tilde{d}_k}, \boldsymbol{W}_i^K\in\mathbb{R}^{d_k\times \tilde{d}_k}, \boldsymbol{W}_i^V\in\mathbb{R}^{d_v\times \tilde{d}_v}$
那么在整个模型中，是如何使用attention的呢？如下图，首先在编码器到解码器的地方使用了多头attention进行连接，K，V，Q分别是编码器的层输出（这里K=V）和解码器中都头attention的输入。其实就和主流的机器翻译模型中的attention一样，利用解码器和编码器attention来进行翻译对齐。然后在编码器和解码器中都使用了多头自注意力self-attention来学习文本的表示。Self-attention即K=V=Q，例如输入一个句子，那么里面的每个词都要和该句子中的所有词进行attention计算。目的是学习句子内部的词依赖关系，捕获句子的内部结构。
那么在整个模型中，是如何使用attention的呢？如下图，首先在编码器到解码器的地方使用了多头attention进行连接，K，V，Q分别是编码器的层输出（这里K=V）和解码器中都头attention的输入。其实就和主流的机器翻译模型中的attention一样，利用解码器和编码器attention来进行翻译对齐。然后在编码器和解码器中都使用了多头自注意力self-attention来学习文本的表示。Self-attention即K=V=Q，例如输入一个句子，那么里面的每个词都要和该句子中的所有词进行attention计算。目的是学习句子内部的词依赖关系，捕获句子的内部结构。

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/Attention_5.png"/>
</div>

对于使用自注意力机制的原因，论文中提到主要从三个方面考虑（每一层的复杂度，是否可以并行，长距离依赖学习），并给出了和RNN，CNN计算复杂度的比较。可以看到，如果输入序列n小于表示维度d的话，每一层的时间复杂度self-attention是比较有优势的。当n比较大时，作者也给出了一种解决方案self-attention（restricted）即每个词不是和所有词计算attention，而是只与限制的r个词去计算attention。在并行方面，多头attention和CNN一样不依赖于前一时刻的计算，可以很好的并行，优于RNN。在长距离依赖上，由于self-attention是每个词和所有词都要计算attention，所以不管他们中间有多长距离，最大的路径长度也都只是1。可以捕获长距离依赖关系。
#### 3. Position Embedding
但是仅依靠以上描述构建出来的Attention模型其实**并不能捕捉到序列的顺序**。如果将$\boldsymbol{K},\boldsymbol{V}$按行打乱顺序，相当于打乱了句子中的词序，那么通过上面的模型得到的结果还是一样的。为了解决这个问题，Google提出了Position Embedding,即“位置向量”(实际上，这种位置向量在先前的很多论文中也有，但google提出的有所不同)。其将序列的每个位置进行编号，一个编号对应一个向量，通过结合位置向量和词向量，就给每个词都引入了一定的位置信息，这样Attention就可以分辨出不同位置的词了。Google提出的Position Embedding在于：
1. 在RNN、CNN模型中都曾出现过Position Embedding，但由于RNN、CNN本身就能捕捉到位置信息，因此Position Embedding只是起到一个辅助的作用。
但是在Google提出的这个纯Attention模型中，Position Embedding是位置信息的唯一来源，因此它是模型的核心成分之一，并非仅仅是简单的辅助手段。
2. 在以往的Position Embedding中，基本都是根据任务训练出来的向量。而Google直接给出了一个构造Position Embedding的公式：

<div align=center>
<img src="https://raw.githubusercontent.com/AMANKB/NLPnotes/master/BERT-notes/images/position_embedding.png"/>
</div>

这里的意思是将id为$\boldsymbol{p}$的位置映射为一个$\boldsymbol{d_pos}$维的位置向量，这个向量的第i个元素的数值就是$\boldsymbol{PE_(2i,p)}$.
3. 论文中还提到选择这样的一个位置向量公式的重要原因在于：**Position Embedding尽管本身是代表的是一个绝对位置的信息，但是它能提供表达相对位置信息的可能性。**这一点可以从正弦的三角函数公式看到，$\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$
和$\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta$,这表明位置$p+k$的向量可以表示成位置$p$的向量的线性变换。


