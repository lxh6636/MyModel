import sys
import math
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('utils')
from proj_adaptive_softmax import ProjectedAdaptiveLogSoftmax
from log_uniform_sampler import LogUniformSampler, sample_logits


class MHPooling(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        "Take in model size and number of heads."
        super(MHPooling, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # auto-regressive
        attn_shape = (1, 3000, 3000)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        self.mask = (torch.from_numpy(subsequent_mask) == 0).unsqueeze(1).cuda()

    def forward(self, x):
        "Implements Figure 2"

        nbatches, seq_len, d_model = x.shape

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (x, x, x))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=self.mask[:, :, :seq_len, :seq_len],
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure

        nn.LSTM(input_dim, hidden_dim, batch_first=True)
        input_dim：输入词向量的维度=d_model
        hidden_dim：隐藏层神经元个数，或者也叫输出的维度，自己设置多少个神经元
        batch_first：输入输出的第一维是否为 batch_size，默认值 False

        nn.RNN:
        input_shape=[批量大小,时间步数,特征维度]
        output_shape=[批量大小,时间步数,隐藏单元个数]
        """

        self.ksize = ksize
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(output_dim, output_dim, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(output_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(output_dim, output_dim, batch_first=True)

        # nn.Sequential()用来包装层，把几个层包装成一块的样子
        self.output = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU())

        # To speed up
        # j从ksize-1到10000遍历，步长为1；i从j-(ksize-1)到j+1遍历，步长为1
        # idx=[]:遍历得到的所有i都放在[]中
        idx = [i for j in range(self.ksize - 1, 10000, 1) for i in range(j - (self.ksize - 1), j + 1, 1)]
        self.select_index = torch.LongTensor(idx).cuda()
        self.zeros = torch.zeros((self.ksize - 1, input_dim)).cuda()  # size:[ksize-1,input_dim]

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x)  # b x seq_len x ksize x d_model
        batch, l, ksize, d_model = x.shape
        # 去掉窗口的维度ksize
        h = self.rnn(x.view(-1, self.ksize, d_model))[0][:, -1, :]
        return h.view(batch, l, d_model)

    def get_K(self, x):
        batch_size, l, d_model = x.shape
        # zeros:[ksize-1,input_dim]
        # zeros.unsqueeze(0):[1,ksize-1,input_dim]
        # zeros.unsqueeze(0).repeat(batch_size, 1, 1):[batch_size*1,1*(ksize-1),1*input_dim]
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)  # dim=1表示zeros和x在第2个维度上拼接,维度为[batch_size,ksize-1+l,d_model]
        # x的第二个维度上取index_select的索引0-ksize*l对应于
        key = torch.index_select(x, 1, self.select_index[:self.ksize * l])
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key


class LocalRNNLayer(nn.Module):
    "Encoder is made up of attconv and feed forward (defined below)"

    def __init__(self, input_dim, output_dim, rnn_type, ksize, dropout):
        super(LocalRNNLayer, self).__init__()
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, dropout)
        self.connection = SublayerConnection(output_dim, dropout)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.connection(x, self.local_rnn)
        return x


class Block(nn.Module):
    """
    One Block，包括LocalRNN、MHAttention、layernorm、FFN
    """

    def __init__(self, input_dim, output_dim, rnn_type, ksize, N, h, dropout):
        super(Block, self).__init__()
        self.layers = clones(
            LocalRNNLayer(input_dim, output_dim, rnn_type, ksize, dropout), N)
        self.connections = clones(SublayerConnection(output_dim, dropout), 2)
        self.pooling = MHPooling(input_dim, h, dropout)  # 计算多头注意力
        self.feed_forward = PositionwiseFeedForward(input_dim, dropout)

    def forward(self, x):
        n, l, d = x.shape
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.connections[0](x, self.pooling)
        x = self.connections[1](x, self.feed_forward)
        return x

## 8.位置嵌入
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb

        # torch.arange(start=0, end, step=1, out=None)
        # 1/10000^(i/2)，i以2为间隔
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb)) # torch.Size([demb/2])

        #向模块添加持久缓冲区
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        # pos_seq：序列位置向量
        # torch.ger(a, b):b中所有元素乘以a中元素进行扩维,pytorch1.7版本后torch.outer取代torch.ger
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)  # torch.Size([pos_seq, demb/2])
        # dim=0:按列堆起来;dim=1:按行堆起来;dim=-1:最后一个维度进行操作,这里sin在前cos在后
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)  # torch.Size([pos_seq, demb])

        if bsz is not None:
            # 增加一个维度,得到的相对位置嵌入向量维度：torch.Size([pos_seq, bsz, demb])，其中demb=d_model
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

## 7.FFN层
class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        # d_inner:FFN层内部映射所使用的维度
        self.d_inner = d_inner
        self.dropout = dropout

        # FFN层实现
        # 第一个网络模块的输出传入第二个网络模块作为输入，按照顺序依次计算并传播，直到nn.Sequential()里的最后一个模块输出结果
        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        # inp:输入的隐藏状态H
        # 如果pre-lnorm，则先LN再残差，否则先残差再LN
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            # FFN(LN(H))
            core_out = self.CoreNet(self.layer_norm(inp))

            # residual connection
            # output = FFN(LN(H))+H  则先LN再残差
            output = core_out + inp
        else:
            # positionwise feed-forward
            # FFN(H)
            core_out = self.CoreNet(inp)

            # residual connection + layer normalization
            # output = LN(FFN(H) + H)  先残差再LN
            output = self.layer_norm(inp + core_out)

        return output

#
class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm=False):

        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head  # 注意力头的个数
        self.d_model = d_model  # 模型的隐层维度
        self.d_head = d_head  # 每个头的隐层维度(dk)
        self.dropout = dropout

        # qkv_net是用于计算query、key和value变换的参数矩阵Wq , Wk , Wv
        # Linear(in_features, out_features, bias):输入二维张量大小in_features([batch_size,size]中的size)，输出二维张量的大小out_features([batch_size,out_size]中的out_size)
        self.q_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.kv_net = nn.Linear(d_model, 2 * n_head * d_head, bias=False)

        # 如nn.Dropout(p = 0.3):表示每个神经元有0.3的可能性不被激活,只能用在训练部分而不能用在测试部分
        # Dropout一般用在全连接神经网络映射层之后，如代码的nn.Linear(20, 30)之后
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # o_net是用于将所有注意力头的结果拼接后再变换到模型维度的参数矩阵
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        # layer_norm是LayerNormalization层
        self.layer_norm = nn.LayerNorm(d_model)

        # 根号dk
        self.scale = 1 / (d_head ** 0.5)

        # 把LayerNorm放在Residual过程前就是pre_lnome(比post_lnorm效果好很多)，若放在Residual之后就是post_lnorm
        self.pre_lnorm = pre_lnorm

    def forward(self, h, attn_mask=None, mems=None):
        """

        :param h:隐藏状态,torch.Size([hlen, batch_size, n_head, d_head]),hlen:隐藏层维度也就是d_model
        :param attn_mask:
        :param mems:memory
        :return:
        """

        if mems is not None:
            # 有缓存就把缓存与隐藏状态进行拼接，按行堆起来，有c:torch.Size([mlen+hlen, batch_size, n_head, d_head])
            c = torch.cat([mems, h], 0)
        else:
            c = h

        if self.pre_lnorm:
            c = self.layer_norm(c)

        head_q = self.q_net(h)
        head_k, head_v = torch.chunk(self.kv_net(c), 2, -1)

        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = h + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        return output

## 6.Multi-Head-Attn 层的父类
class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        # 生成Q、K、V向量的线性映射
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        # 整体的dropout
        self.drop = nn.Dropout(dropout)

        # attention的dropout
        self.dropatt = nn.Dropout(dropatt)

        # 将多个头的向量拼接映射到d_model，是最后一个输出线性映射
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        # 层级的正则化
        self.layer_norm = nn.LayerNorm(d_model)

        # 一个放缩因子
        self.scale = 1 / (d_head ** 0.5)

        # 输入层正则化
        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        """

        :param h:
        :param w: 上一层输入
        :param left:
        :return:
        """
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        # 上三角(包括多角形)元素保留，其余为0
        mask[:m,:m] = torch.triu(mask[:m,:m])
        # 若m=3，则mask[-m:,-m:]:mask的-3,-2,-1行及-3,-2,-1列所构成的矩阵
        # 下三角(包括多角形)元素保留，其余为0
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            # 对0维进行反转，即行逆向排序
            return mask.flip(0)


    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    # 转化相对位置：先添加当前词位置为0，在向前推qlen个位置
    # 该方法没有数据的拷贝，全部都是view操作，因此更高效
    def _rel_shift(self, x, zero_triu=False):
        # x.size()=[q, k, bzs, n_head]
        # x的第一列padding
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        # padding和x按列拼接,得到torch.Size([q, k+1, bzs, n_head])
        x_padded = torch.cat([zero_pad, x], dim=1)
        # torch.Size([q, k+1, bzs, n_head])-->torch.Size([k+1, q, bzs, n_head])
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            # ones为下三角为1其他为0的矩阵，所以这里x保留下三角元素，上三角元素为0
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError  # 子类没有实现父类要求一定要实现的接口

## 5.继承了RelMultiHeadAttn
class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        # *args:接收若干个位置参数，转换成元组tuple形式
        # **kwargs：接收若干个关键字参数，转换成字典dict形式
        # super().__init__():从RelMultiHeadAttn中继承属性(必须有这一句话才能继承属性)
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        # r_net是用于计算relative position embedding变换的参数矩阵Wk,r
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        """

        :param w:上一层当前segment隐藏状态,torch.Size([seq_len, batch_size, d_model]),seq_len为文本输入最大长度
        :param r: pos_emb,torch.Size([pos_seq, bsz, d_model])
        :param r_w_bias:u向量torch.Size([n_head, d_model])
        :param r_r_bias:v向量torch.Size([klen, n_head])
        :param attn_mask:
        :param mems:memory，上一个层的上一个segment的隐藏状态
        :return:
        """

        # qlen=seq_length,rlen=pos_emb_length,bsz=batch_size
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        #### 1.缓存拼接
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                # qkv_net：Q、K、V向量的线性映射Wq,Wk,Wv，torch.Size([d_model, 3*n_head*d_head])
                # 获取上一层当前segment的隐藏状态的多头注意力,torch.Size([qlen, bsz, 3*n_head*d_head])
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)

            # torch.Size([rlen, n_head*d_head])
            r_head_k = self.r_net(r)

            # troch.chunk(w_heads, 3, dim=-1):w_heads在最后一维拆成三份
            # 这里的qkv没有单独作为输入，而是将embedding和Wqkv矩阵相乘之后分段
            # w的q、k、v三者均为torch.Size([qlen, bsz, n_head*d_head])
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            # 将memory从query中剔除，这里key、value中的memory保留
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)

            r_head_k = self.r_net(r)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        # klen=qlen=seq_length
        klen = w_head_k.size(0)

        #### 2.多头操作
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### 3.计算注意力分数
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head

        # 把维度为ibnd的张量与维度为ijbn的张量相乘并转化为ijbn的张量,得到公式(a)+(c)
        # AC=w_head_q*w_head_k+r_w_bias*w_head_k=rw_head_q*w_head_k
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias

        # 公式(b)+(d)
        # BD=w_head_q*r_head_k+r_r_bias*r_head_k=rr_head_q*r_head_k
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        attn_score = AC + BD                                                    # qlen x klen x bsz x n_head
        attn_score.mul_(self.scale)

        #### 4.注意力进行mask(encoder用pad mask,decoder用seq mask和pad mask)
        # 通过PADDING MASK的操作，补全位置上的值成为负无穷，这样的话，经过Softmax层的时候，这些位置上的概率就是0
        # any():判断给定的可迭代参数是否全为False，是则返回False，否则返回True
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                # attn_score.masked_fill(mask, value):mask中取值为True位置对应于attn_score的相应位置用value填充
                # mask是一个tensor，元素是布尔值(mask与attn_score的维度要保持一致，所以这里用的attn_mask[None,:,:,None])
                # value是要填充的值，这里用的-float('inf')表示负无穷
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        #### 5.计算注意力分布
        # 对attn_score的1维度进行归一化，即行和为1
        attn_prob = F.softmax(attn_score, dim=1)
        # 默认不保留概率p为0.5，tensor非零的元素会变为原来的1/(1-p)倍数
        attn_prob = self.dropatt(attn_prob)                                       # qlen x klen x bsz x n_head

        #### 6.计算头向量
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))         # qlen x bsz x n_head x d_head

        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)        # qlen x bsz x (n_head*d_head)

        ##### 7.线性映射
        # o.net():将多个头的向量拼接映射到d_model nn.Linear(n_head*d_head, d_model)
        attn_out = self.o_net(attn_vec)                                           # qlen x bsz x d_model
        attn_out = self.drop(attn_out)                                            # qlen x bsz x d_model

        if self.pre_lnorm:
            # residual connection
            # w:上一层当前segment的隐藏状态
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class RelLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

    def forward(self, w, r_emb, r_w_bias, r_bias, attn_mask=None, mems=None):
        # r_emb: [klen, n_head, d_head], used for term B
        # r_w_bias: [n_head, d_head], used for term C
        # r_bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1)

        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        if klen > r_emb.size(0):
            r_emb_pad = r_emb[0:1].expand(klen-r_emb.size(0), -1, -1)
            r_emb = torch.cat([r_emb_pad, r_emb], 0)
            r_bias_pad = r_bias[0:1].expand(klen-r_bias.size(0), -1)
            r_bias = torch.cat([r_bias_pad, r_bias], 0)
        else:
            r_emb = r_emb[-klen:]
            r_bias = r_bias[-klen:]

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias[None]                                   # qlen x bsz x n_head x d_head

        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head
        B_ = torch.einsum('ibnd,jnd->ijbn', (w_head_q, r_emb))                  # qlen x klen x bsz x n_head
        D_ = r_bias[None, :, None]                                              # 1    x klen x 1   x n_head
        BD = self._rel_shift(B_ + D_)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(DecoderLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

class RelLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout,
                                         **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output
## 3.解码层
class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)  # 4
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout,
                                     pre_lnorm=kwargs.get('pre_lnorm'))  # 7

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output

## 2. 词嵌入
class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 sample_softmax=False):
        """

        :param n_token: 28
        :param d_embed: 500
        :param d_proj: 500
        :param cutoffs: []
        :param div_val: 1
        :param sample_softmax: False
        """
        super(AdaptiveEmbedding, self).__init__()

        self.n_token = n_token  # 28
        self.d_embed = d_embed  # 500

        self.cutoffs = cutoffs + [n_token]  # [28]
        self.div_val = div_val  # 1
        self.d_proj = d_proj  # 500

        self.emb_scale = d_proj ** 0.5  # 500^0.5

        self.cutoff_ends = [0] + self.cutoffs  # [0, 28]

        self.emb_layers = nn.ModuleList()  # Embedding(28, 500)
        self.emb_projs = nn.ParameterList()
        if div_val == 1:
            self.emb_layers.append(
                nn.Embedding(n_token, d_embed, sparse=sample_softmax>0)
            )
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_embed)))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                d_emb_i = d_embed // (div_val ** i)
                self.emb_layers.append(nn.Embedding(r_idx-l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.Tensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed  = F.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())
            inp_flat = inp.view(-1)
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj],
                dtype=param.dtype, device=param.device)
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]

                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                indices_i = mask_i.nonzero().squeeze()

                if indices_i.numel() == 0:
                    continue

                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                emb_i = self.emb_layers[i](inp_i)
                emb_i = F.linear(emb_i, self.emb_projs[i])

                emb_flat.index_copy_(0, indices_i, emb_i)

            embed = emb_flat.view(*inp.size(), self.d_proj)

        embed.mul_(self.emb_scale)

        return embed

## 1.记忆Transformer语言模型
class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1, tie_projs=[False], pre_lnorm=False,
                 tgt_len=None, ext_len=None, mem_len=None,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, attn_type=0, clamp_len=-1,
                 sample_softmax=-1):
        super(MemTransformerLM, self).__init__()
        self.n_token = n_token  # 28

        d_embed = d_model if d_embed is None else d_embed
        self.d_embed = d_embed  # d_embed=500
        self.d_model = d_model  # d_model=500
        self.n_head = n_head  # n_head=10
        self.d_head = d_head  # d_head=50

        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs,
                                          div_val=div_val)  # 2

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer  # n_layer=12

        self.tgt_len = tgt_len  # 需要predict的token的数目 tgt_len=70 val_tgt_len=50
        self.mem_len = mem_len  # 保留的前一个多头的长度k_length，mem_len不断扩大
        self.ext_len = ext_len  # 训练的预测长度tgt_len-评估的预测长度val_tgt_len

        # 若不用缓存，mem_len=0，max_klen=tgt_len+ext_len
        # 若用缓存，ext_len=0，max_klen=tgt_len+mem_len
        self.max_klen = tgt_len + ext_len + mem_len

        self.attn_type = attn_type  # 0

        self.layers = nn.ModuleList()

        if attn_type == 0: # 文中的Transformer-XL
            for i in range(n_layer):
                self.layers.append(
                    RelPartialLearnableDecoderLayer(  # 3
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type == 1: # learnable embeddings
            for i in range(n_layer):
                self.layers.append(
                    RelLearnableDecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )
        elif attn_type in [2, 3]: # absolute embeddings
            for i in range(n_layer):
                self.layers.append(
                    DecoderLayer(
                        n_head, d_model, d_head, d_inner, dropout,
                        dropatt=dropatt, pre_lnorm=pre_lnorm)
                )

        self.sample_softmax = sample_softmax  # sample_softmax=-1
        # use sampled softmax
        if sample_softmax > 0:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
            self.tie_weight = tie_weight
            self.sampler = LogUniformSampler(n_token, sample_softmax)

        # use adaptive softmax (including standard softmax)
        else:
            self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model,
                                                    cutoffs, div_val=div_val)

            if tie_weight:  # tie_weight=True
                for i in range(len(self.crit.out_layers)):
                    self.crit.out_layers[i].weight = self.word_emb.emb_layers[i].weight

            if tie_projs:  # tie_projs=False
                for i, tie_proj in enumerate(tie_projs):
                    if tie_proj and div_val == 1 and d_model != d_embed:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[0]
                    elif tie_proj and div_val != 1:
                        self.crit.out_projs[i] = self.word_emb.emb_projs[i]

        self.same_length = same_length  # same_length=False
        self.clamp_len = clamp_len   # clamp_len=-1

        # 得到相关位置嵌入
        self._create_params()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        """
        得到相对位置嵌入
        函数调用：self._create_params()
        :return:
        """
        if self.attn_type == 0: # default attention
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        elif self.attn_type == 1: # learnable
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))
            self.r_w_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.n_head, self.d_head))
            self.r_bias = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head))
        elif self.attn_type == 2: # absolute standard
            self.pos_emb = PositionalEmbedding(self.d_model)
        elif self.attn_type == 3: # absolute deeper SA
            self.r_emb = nn.Parameter(torch.Tensor(
                    self.n_layer, self.max_klen, self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        """
        初始化 0-n_layer层的上段缓存的上文信息(mems)，mems只是上一个段的缓存，不是整个段的缓存
        :return: 返回一个list，每个元素是一个tensor，即[tensor([]), tensor([]), tensor([])，...]，一共n_layer个元素
        """
        if self.mem_len > 0:
            mems = []
            # next：通过调用它的next()方法从迭代器中检索下一项
            # next(self.parameters())：这里，它返回该类的第一个参数
            param = next(self.parameters())

            # 当前段的每一层的memory都放在列表mems中
            for i in range(self.n_layer+1):
                # empty=temsor([])
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        '''
        mems的更新函数，每次更新，mems被新mems覆盖，mens长度始终保持为mlen
        函数调用：new_mems = self._update_mems(hids, mems, mlen, qlen)
        :param hids: 当前段每层的输出（隐藏状态）,torch.Size([qlen=70, bsz=60, d_model=500]),hids=[hid1,hid2,...]
        :param mems: 当前段每层依赖的memory，即上一段每层隐藏状态，mems=[mem1.mem2,...],len(mems)=len(hids)
        :param qlen: 序列长度
        :param mlen: 当前段依赖的memory长度
        :return:更新之后的memory，即当前段每层的输出成为下一段的memory
        '''
        # does not deal with None
        if mems is None: return None

        # ''的内容是AssertionError输出的内容，即错误的内容
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        '''
            pytorch中，tensor有一个requires_grad参数(默认为false)，如果设置为True，则反向传播时，该tensor就会自动求导,
            with torch.no_grad的作用在该模块下，所有计算得出的tensor的requires_grad都自动设置为False
        '''
        with torch.no_grad():
            new_mems = []

            '''
                已经缓存的长度mlen加上当前片段的长度qlen更新到缓存mems中
                下一段，qlen长的tokens的最后ext_len长度将会用作扩展上下文
                所以，只用缓存mlen+qlen-self.ext_len-self.mem_len到mlen+qlen-self-ext_len的长度
            '''
            # mlen=146,qlen=0
            # end_idx=146+max(0,0-0)=146
            # end_idx=146+max(0,150)=296
            # 减去下一次用于扩展上下文的词的隐藏层状态向量
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)

            # beg_idx=max(0,146-150)=0
            # beg_idx=max(0,296-150)=146
            beg_idx = max(0, end_idx - self.mem_len)

            for i in range(len(hids)):
                # mems和hids行拼接，列不变
                cat = torch.cat([mems[i], hids[i]], dim=0)
                # cat[beg_idx:end_idx].detach()：使cat[beg_idx:end_idx]的参数不进行反向传播，其他参数可以
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def _forward(self, dec_inp, mems=None):
        """
        # 函数调用： hidden, new_mems = self._forward(data, mems=mems)
        :param dec_inp:data-->torch.Size([qlen=70, bsz=60])
        :return:core_out(最上层的隐藏层状态输出), new_mems
        """
        qlen, bsz = dec_inp.size()  # qlen=70, bsz=60

        word_emb = self.word_emb(dec_inp)  # torch.Size([70, 60, 500])

        # mlen:当前段第一层缓存的memory(上一段最后一层的隐藏状态)的长度
        # mems[0]:第一层的缓存 ; .size(0):缓存长度
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:  # same_length=False
            # 词嵌入置为1
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            # same_length=True，则只会关注当前词在内的前len(segment)词
            dec_attn_mask = (torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            # same_length=False，则只是mask当前segment所在词之后的词(右半个矩阵)，而对mem中的信息全部使用(左半个矩阵)
            # 如果diagonal为正数n，则输入矩阵置主对角线n行以上的元素为1
            # dec_attn_mask:torch.Size([qlen, mlen, 1])
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).byte()[:,:,None]

        hids = []  # 记录每层的输入向量

        if self.attn_type == 0: # default=0 文中的Transformer-XL
            # arange(start=klen-1,stop=-1,step=-1)
            # pos_seq=[klen-1,....,0]，torch.Size([klen])
            pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            # clamp_len之后使用相同的位置嵌入
            if self.clamp_len > 0: # -1
                pos_seq.clamp_(max=self.clamp_len)

            # 对位置序列进行位置嵌入 pos_emb:torch.Size([klen,bsz,d_model])
            pos_emb = self.pos_emb(pos_seq)  # self.pos_emb = PositionalEmbedding(self.d_model)

            core_out = self.drop(word_emb)
            pos_emb = self.drop(pos_emb)

            # 将最底层输入放入hids中
            hids.append(core_out)

            # decoder
            # self.layers=nn.ModuleList() self.layers.append(RelPartialLearnableDecoderLayer)
            # self.layers中放入了n_layer层的解码器，包括(自注意力、交叉注意力、FFNf)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                # 每一层解码器的输出的隐藏状态放入hids中
                hids.append(core_out)

        elif self.attn_type == 1: # learnable
            core_out = self.drop(word_emb)
            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                if self.clamp_len > 0:
                    r_emb = self.r_emb[i][-self.clamp_len :]
                    r_bias = self.r_bias[i][-self.clamp_len :]
                else:
                    r_emb, r_bias = self.r_emb[i], self.r_bias[i]

                mems_i = None if mems is None else mems[i]
                core_out = layer(core_out, r_emb, self.r_w_bias[i],
                        r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 2: # absolute
            pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device,
                                   dtype=word_emb.dtype)
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq)

            core_out = self.drop(word_emb + pos_emb[-qlen:])

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and i == 0:
                    mems_i += pos_emb[:mlen]
                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)
        elif self.attn_type == 3:
            core_out = self.drop(word_emb)

            hids.append(core_out)
            for i, layer in enumerate(self.layers):
                mems_i = None if mems is None else mems[i]
                if mems_i is not None and mlen > 0:
                    cur_emb = self.r_emb[i][:-qlen]
                    cur_size = cur_emb.size(0)
                    if cur_size < mlen:
                        cur_emb_pad = cur_emb[0:1].expand(mlen-cur_size, -1, -1)
                        cur_emb = torch.cat([cur_emb_pad, cur_emb], 0)
                    else:
                        cur_emb = cur_emb[-mlen:]
                    mems_i += cur_emb.view(mlen, 1, -1)
                core_out += self.r_emb[i][-qlen:].view(qlen, 1, -1)

                core_out = layer(core_out, dec_attn_mask=dec_attn_mask,
                                 mems=mems_i)
                hids.append(core_out)

        core_out = self.drop(core_out)  # 最上层的隐藏层状态输出

        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def forward(self, data, target, *mems):
        # nn.DataParallel does not allow size(0) tensors to be broadcasted.
        # So, have to initialize size(0) mems inside the model forward.
        # Moreover, have to return new_mems to allow nn.DataParallel to piece
        # them together.

        # 得到一个初始化的new_mems，mems=[tensor([]), tensor([]), tensor([])，...]
        if not mems: mems = self.init_mems()

        tgt_len = target.size(0)  # train:tgt_len=70 val/test:tgt_len=50

        # 得到decoder最后一层输出的隐藏状态hidden和更新后的new_mems
        # hidden:[klen,bsz,d_model]  klen=mlen+qlen
        hidden, new_mems = self._forward(data, mems=mems)

        # pred_hid:[tgt_len,bsz,d_model]
        pred_hid = hidden[-tgt_len:]

        if self.sample_softmax > 0 and self.training:  # sample_sotfmax=-1
            assert self.tie_weight
            logit = sample_logits(self.word_emb,
                self.out_layer.bias, target, pred_hid, self.sampler)
            loss = -F.log_softmax(logit, -1)[:, :, 0]
        else:
            # self.crit=ProjectedAdaptiveLogSoftmax(),得到nll,使用的adaptive softmax
            loss = self.crit(pred_hid.view(-1, pred_hid.size(-1)), target.view(-1))
            loss = loss.view(tgt_len, -1)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")  # name 'args' is not defined

    B = 4  # batch_size
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20  # 20个batch
    args.n_token = 10000

    import data_utils  # 数据集

    # data=[2880]
    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]  # [5000]
    tie_projs = [False] + [True] * len(cutoffs)  # [False,True...True]

    # div_val、d_embed分别为[1, 200], [1, 100], [2, 200], [2 ,100]输入到模型中
    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            # to(device):将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True,
                            d_embed=d_embed, div_val=div_val,
                            tie_projs=tie_projs, pre_lnorm=True,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                            cutoffs=cutoffs, attn_type=0).to(device)

            # model.parameters():获取模型参数;p.numel():获取参数的元素个数
            print(sum(p.numel() for p in model.parameters()))
            '''
                数据说明：
                    data=[740,4],每一列数一个句子，batch_size=4;
                    每次截取36词作为一个样本，训练语言模型
             '''
            mems = tuple()
            # enumerate(diter)：将一个可遍历的数据对象(diter)组合为一个索引序列，用于循环
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]