import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        # 先增加一个维度，相当于复制再扩张，K_expand:维度为[32,8,96,96,64]
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)

        # index_sample:维度为[L_Q,sample_k]的随机数，随机数范围是0到L_K-1
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q

        # K_sample:[32, 8, 96, 25, 64]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]

        # Q.unsqueeze(-2):[32, 8, 96, 1, 64]
        # K_sample.transpose(-2, -1):[32, 8, 96, 64, 25]
        # Q_K_sample:[32, 8, 96, 25]，96个Q和25个K直接的关系
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        # 96个Q中，每一个Q选出和其他K关系最大的值(Q_K_sample.max(-1))，再计算与均匀分布的差异
        # M:[32, 8, 96]
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)

        # 从96个Q的评分中找出n_top=25个最高得分，返回值1表示要得到索引
        # M_top:[32, 8, 25]
        M_top = M.topk(n_top, sorted=False)[1]

        # Q_reduce:[32, 8, 25, 64]，已经采样完25个特征最明显的Q了。K还是一样保持了96个
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)

        # Q_k:[32, 8, 25, 96]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top

    # 对V矩阵进行处理
    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:  # mask_flag=True
            # V_sum = V.sum(dim=-2)   解码器的稀疏自注意力用的是求和V
            V_sum = V.mean(dim=-2)  # 编码器的稀疏自注意力用的是平均V，把所有的V都初始化为我们的平均值，一直平庸
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            # 如cumsum(a,dim=0):第一行不动，把上一行加到下一行，一直累加到最后一行
            # context:[32, 8, 96, 64]
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        """

        :param context_in:输入的上一层上下文表示
        :param V:[32, 8, 96, 64]
        :param scores: foward中输入的是scores_top=Q_K :[32,8,25,96]
        :param index: index=M_top:[32, 8, 25]
        :param L_Q:96
        :param attn_mask:
        :return:
        """
        B, H, L_V, D = V.shape

        # 自注意力做掩码
        if self.mask_flag:  # mask_flag=True
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # attn=Softmax(QK/(d^0.5)) :[32, 8, 25, 96]
        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)

        # context_in=attn*V，再把维度调整与输入上下文向量一致
        # 输入的context_in:[32, 8, 96, 64]
        # 输出的context_in:[32, 8, 96, 64]
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)  # 这里的V是求平均之后的V

        # 只输出context，或者输出context+多头注意力
        if self.output_attention:  # output_attention=False
            # attns:[32, 8, 96, 96]，每个元素值为1/96
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            # attns:[32,8,96,96],是[32,8,25,96]的attn与每个元素值为1/96,维度为[32,8,71,96]的拼接
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # 筛选出明显特征的Q，找出Q的代表;np.ceil():得到大于等于该值的最小整数;item():一个元素张量可以用x.item()得到元素值
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K  # U_part=25 key里要选的个数
        u = u if u<L_Q else L_Q  # u=25

        # scores_top=Q_k, index=M_top
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor 排除维度D带来的干扰
        scale = self.scale or 1./sqrt(D)  # self.scale=None
        if scale is not None:
            scores_top = scores_top * scale
        # 初始化上下文向量
        context = self._get_initial_context(values, L_Q)
        # 用25个Q矩阵更新上下文向量
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix  # H和d_model是否合并

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape  # B:32 L:96
        _, S, _ = keys.shape  # S:96
        H = self.n_heads  # H:8

        # 利用view形成多头
        queries = self.query_projection(queries).view(B, L, H, -1)  # [32, 96, 8, 64]
        keys = self.key_projection(keys).view(B, S, H, -1)  # [32, 96, 8, 64]
        values = self.value_projection(values).view(B, S, H, -1)  # [32, 96, 8, 64]

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:  # Informer类和InformerStack类中mix=False
            # contiguous()强制拷贝一份tensor，两个out没有如何关系
            # out:[32,96,8,64]
            out = out.transpose(2,1).contiguous()
        # out:[32,96,8*64=512]
        out = out.view(B, L, -1)

        # out_projection()：经过一个全连接层，全连接只针对最后一维特征进行全连接
        return self.out_projection(out), attn
