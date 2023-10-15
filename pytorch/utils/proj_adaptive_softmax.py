from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

CUDA_MAJOR = int(torch.version.cuda.split('.')[0])  # 11
CUDA_MINOR = int(torch.version.cuda.split('.')[1])  # 3

class ProjectedAdaptiveLogSoftmax(nn.Module):
    """
    self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val=div_val)
    将词汇表按照频率分成不同的 cluster，其中len(cutoffs)就是cluster的数量
    每个cluster都有不同维度的embedding矩阵
    在每个cluster内先将隐藏层状态向量经过一个映射d_proj，将其维度变成与该cluster的embedding矩阵相同的维度
    再经过对应的embedding矩阵将其映射到词汇表得到logit(非规范化概率)，在得到对数概率
    """
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1,
                 keep_order=False):
        super(ProjectedAdaptiveLogSoftmax, self).__init__()

        self.n_token = n_token  # 28
        self.d_embed = d_embed  # 500
        self.d_proj = d_proj  # d_proj=d_model=500

        self.cutoffs = cutoffs + [n_token]  # [28]
        self.cutoff_ends = [0] + self.cutoffs  # [0, 28]
        # 组与组之间d_embed的下降系数，d_embed=d_embed//div_val，词嵌入维度保存不变
        self.div_val = div_val  # 1

        self.shortlist_size = self.cutoffs[0]  # 28
        # 词汇表分类数即低频词的cluster
        self.n_clusters = len(self.cutoffs) - 1  # 0
        # 根节点元素个数
        self.head_size = self.shortlist_size + self.n_clusters  # 28

        # 为实现根部的映射而设计的参数
        if self.n_clusters > 0:
            # Parameter containing: [torch.FloatTensor of size self.n_clusters*self.d_embed]
            self.cluster_weight = nn.Parameter(torch.zeros(self.n_clusters, self.d_embed))
            # Parameter containing: [torch.FloatTensor of size self.n_clusters]
            self.cluster_bias = nn.Parameter(torch.zeros(self.n_clusters))

        self.out_layers = nn.ModuleList()
        self.out_projs = nn.ParameterList()

        if div_val == 1:  # 词嵌入维度保存不变即full softmax的情况
            for i in range(len(self.cutoffs)):
                if d_proj != d_embed:  # 词嵌入维度和映射维度不等，要把映射维度转化为嵌入维度，添加一个映射层
                    # ParameterList((0): Parameter containing: [torch.FloatTensor of size d_proj*d_embed])
                    self.out_projs.append(
                        nn.Parameter(torch.Tensor(d_proj, d_embed))
                    )
                else:  # 词嵌入维度和映射维度相等，不需要映射层
                    self.out_projs.append(None)

            # ModuleList((0): Linear(in_features=d_embed, out_features=n_token, bias=True))
            # 输出隐藏向量经过一个线性层，把词嵌入维度转化为词汇表维度
            self.out_layers.append(nn.Linear(d_embed, n_token))
        else:
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i+1]
                # 每个cluster都有一个映射矩阵，这是映射维度，比如1024,512,256,128
                d_emb_i = d_embed // (div_val ** i)

                self.out_projs.append(
                    nn.Parameter(torch.Tensor(d_proj, d_emb_i))
                )

                self.out_layers.append(nn.Linear(d_emb_i, r_idx-l_idx))

        self.keep_order = keep_order  # False

    def _compute_logit(self, hidden, weight, bias, proj):
        """
        将hidden经过线性映射proj得到与weight相同维度，再经过(weight,bias)映射得到预测向量
        :param hidden:
        :param weight:
        :param bias:
        :param proj:
        :return:
        """
        # d_proj等于d_embed
        if proj is None:
            logit = F.linear(hidden, weight, bias=bias)

        # d_proj不等于d_embed
        else:
            # if CUDA_MAJOR <= 9 and CUDA_MINOR <= 1:
            proj_hid = F.linear(hidden, proj.t().contiguous())
            logit = F.linear(proj_hid, weight, bias=bias)
            # else:
            #     logit = torch.einsum('bd,de,ev->bv', (hidden, proj, weight.t()))
            #     if bias is not None:
            #         logit = logit + bias

        return logit

    def forward(self, hidden, target, keep_order=False):
        '''
            hidden、target都是mem_transformer.py里的变量
            hidden : pred_hid.view(-1, pred_hid.size(-1))-->[tgt_len*bsz, d_proj]
            target : target.view(-1)-->[tgt_len*bsz]
        '''

        if hidden.size(0) != target.size(0):  # tgt_len*bsz
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        if self.n_clusters == 0:  # full softmax的情况
            logit = self._compute_logit(hidden, self.out_layers[0].weight,
                                        self.out_layers[0].bias, self.out_projs[0])
            # 负对数似然损失
            nll = -F.log_softmax(logit, dim=-1) \
                    .gather(1, target.unsqueeze(1)).squeeze(1)
        else:
            # 构建权重和偏差
            weights, biases = [], []
            for i in range(len(self.cutoffs)):
                if self.div_val == 1:  # 每个cluster的映射维度d_embed一样，如512、512、512、512，只有一个线性层用来映射
                    # 获得第i组cluster的左索引l_idx，右索引r_idx
                    l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                    # 获得线性层第i组cluster对应的权重
                    weight_i = self.out_layers[0].weight[l_idx:r_idx]
                    # 获得线性层第i组cluster对应的偏差
                    bias_i = self.out_layers[0].bias[l_idx:r_idx]
                else:  # 每个cluster的映射维度d_embed不一样，如1024、512、256、128
                    # 每组cluster的d_embed不一样，每组cluster对应每一个线性层操作，每组cluster的权重和偏差从每个线性层中获取
                    weight_i = self.out_layers[i].weight
                    bias_i = self.out_layers[i].bias

                # 拼接根部的矩阵(全0，维度为n_clusters*d_embed的矩阵)，第一组cluster对应的权重和偏差才需要拼接
                # 若len(cutoffs)=4，则n-clusters=3，第一个cluster存放高频词，后三个cluster存放低频词
                # 记每一个cluster分别为v1,v2,v3,v4，v1的维度为r_idx-l_idx，v2,v3,v4作为tail维度为3，根(head)的维度为r_idx-l_idx+3
                if i == 0:

                    # weight_i = torch.cat([r_idx-l_idx, d_embed],[n_clusters, d_embed]，dim=0)
                    # weight:[[r_idx-l_idx+n_clusters, d_embed]]
                    weight_i = torch.cat(
                        [weight_i, self.cluster_weight], dim=0)

                    # bias_i = torch.cat([r_idx-l_idx],[n_clusters, d_embed], dim=0)
                    # bias_i:[r_idx-l_idx+n_clusters]
                    bias_i = torch.cat(
                        [bias_i, self.cluster_bias], dim=0)

                weights.append(weight_i)
                biases.append(bias_i)

            # 根部预测
            head_weight, head_bias, head_proj = weights[0], biases[0], self.out_projs[0]

            # [tgt_len*bsz, head_size]，其中head_size=r_idx-l_idx+n_clusters
            head_logit = self._compute_logit(hidden, head_weight, head_bias, head_proj)
            # [tgt_len*bsz, head_size]
            head_logprob = F.log_softmax(head_logit, dim=1)
            # 记录目标词汇的对数概率的向量 [tgt_len,bsz,d_model]
            nll = torch.zeros_like(target,
                    dtype=hidden.dtype, device=hidden.device)

            offset = 0
            cutoff_values = [0] + self.cutoffs  # [0, 28]
            for i in range(len(cutoff_values) - 1):
                # 区间上下界
                l_idx, r_idx = cutoff_values[i], cutoff_values[i + 1]
                # 取出目标词在该区间的样本索引，一个tensor>a，tensor里>a的元素为True，<=a的元素为False
                # 如果target中某个词在字典索引[l_idx,r_idx)之间，就把这个词设置为True，其他的设置为False
                # mask_i:[tgt_len*bsz]
                mask_i = (target >= l_idx) & (target < r_idx)
                # 对mask_i中的非零(True)元素取索引
                indices_i = mask_i.nonzero().squeeze()

                # 该区间没有词
                if indices_i.numel() == 0:
                    continue

                # 获取第0行，索引为indices_i的元素
                target_i = target.index_select(0, indices_i) - l_idx
                # [在当前区间的词数,head_size]
                head_logprob_i = head_logprob.index_select(0, indices_i)

                if i == 0:
                    # 预测值在头部直接取出对数概率
                    logprob_i = head_logprob_i.gather(1, target_i[:,None]).squeeze(1)
                else:
                    weight_i, bias_i, proj_i = weights[i], biases[i], self.out_projs[i]

                    hidden_i = hidden.index_select(0, indices_i)

                    tail_logit_i = self._compute_logit(hidden_i, weight_i, bias_i, proj_i)
                    tail_logprob_i = F.log_softmax(tail_logit_i, dim=1)

                    # 取出头部第i列的对数概率加上尾部的对数概率
                    logprob_i = head_logprob_i[:, -i] \
                              + tail_logprob_i.gather(1, target_i[:,None]).squeeze(1)

                if (hasattr(self, 'keep_order') and self.keep_order) or keep_order:
                    nll.index_copy_(0, indices_i, -logprob_i)  # 保存原理的顺序
                else:
                    nll[offset:offset+logprob_i.size(0)].copy_(-logprob_i)

                offset += logprob_i.size(0)

        return nll
