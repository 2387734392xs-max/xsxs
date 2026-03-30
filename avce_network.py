import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from Transformer import *
import copy
# 从 models 包中的 base_models 模块导入所有内容
from models.base_models import *
# 从 layers 包中的 hyp_layers 模块导入所有内容
from layers.hyp_layers import *
# 从 geoopt 库中导入 ManifoldParameter 类，用于处理流形参数
from geoopt import ManifoldParameter
import torch
import torch.nn as nn
# 从 scipy 库的 spatial.distance 模块导入 pdist 和 squareform 函数，用于计算距离矩阵
from scipy.spatial.distance import pdist, squareform
from alignment import MultimodalRepresentationAlignment
import numpy as np
import torch.nn.functional as F
import math
import option
# 从 torch.nn.modules.module 模块导入 Module 类，是所有神经网络模块的基类
from torch.nn.modules.module import Module
# 从 torch 库中导入 FloatTensor 类，用于创建浮点型张量
from torch import FloatTensor
# 从 torch.nn.parameter 模块导入 Parameter 类，用于创建可训练的参数
from torch.nn.parameter import Parameter
# 导入 manifolds 模块，可能包含各种流形的定义和操作
import manifolds
# 定义AVCE_Model类，继承自nn.Module，用于构建一个多模态的神经网络模型
class DistanceAdj(Module):
    def __init__(self):
        # 调用父类 Module 的构造函数
        super(DistanceAdj, self).__init__()
        # 创建一个可训练的参数 sigma，初始值为 0.1
        self.sigma = Parameter(FloatTensor(1))
        self.sigma.data.fill_(0.1)
    def forward(self, batch_size, max_seqlen, args):
        # 为了支持批量操作，生成一个从 0 到 max_seqlen-1 的一维数组，并将其形状调整为 (max_seqlen, 1)
        self.arith = np.arange(max_seqlen).reshape(-1, 1)
        # 使用 pdist 函数计算 arith 中各点之间的曼哈顿距离，并将结果转换为 float32 类型
        dist = pdist(self.arith, metric='cityblock').astype(np.float32)
        # 使用 squareform 函数将距离向量转换为距离矩阵，并将其转换为 PyTorch 张量，然后移动到指定设备上
        args = option.parser.parse_args()
        self.dist = torch.from_numpy(squareform(dist)).to(args.device)
        # 对距离矩阵取负指数，其中分母为 e 的指数
        self.dist = torch.exp(-self.dist / torch.exp(torch.tensor(1.)))
        # 在第 0 维上增加一个维度，然后将其复制 batch_size 次
        self.dist = torch.unsqueeze(self.dist, 0).repeat(batch_size, 1, 1).to(args.device)
        return self.dist
class AVCE_Model(nn.Module):
    # 初始化函数，接收参数args
    def __init__(self, args):
        # 调用父类的初始化方法
        super(AVCE_Model, self).__init__()
        # 定义一个深拷贝函数，用于后续复制某些模块
        c = copy.deepcopy
        # 从参数args中获取dropout比例
        dropout = args.dropout
        # 从参数args中获取多头注意力的头数
        nhead = args.nhead
        # 从参数args中获取隐藏层的维度
        hid_dim = args.hid_dim
        # 从参数args中获取前馈网络的维度
        ffn_dim = args.ffn_dim
        args = copy.deepcopy(args)
        self.manifold = getattr(manifolds, args.manifold)()
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1
        # 初始化一个多头注意力模块
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        # 初始化一个位置前馈网络模块
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        # 定义一个全连接层，将1024维的视觉特征映射到hid_dim维
        self.fc_v = nn.Linear(1024, hid_dim)
        # 定义一个全连接层，将128维的音频特征映射到hid_dim维你
        self.fc_a = nn.Linear(128, hid_dim)
        # --- 新增 ---
        # 实例化多模态表示对齐模块
        self.mra = MultimodalRepresentationAlignment()
        self.mra_bias_scaler = nn.Parameter(torch.tensor(1.0))
        self.fc_ew = nn.Linear(128, 256)
        self.fc_ew2 = nn.Linear(32, 128)
        self.fc_aew = nn.Linear(128, 256)
        self.fc_aew2 = nn.Linear(32, 128)
        self.fc_o = nn.Linear(hid_dim, hid_dim)

        # 初始化一个交叉注意力块，其中包含一个Transformer层
        self.cma = CrossAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        # 初始化一个Att_MMIL模块，用于多模态交互
        self.att_mmil = Att_MMIL(hid_dim, args.num_classes,args)
        # 创建 DistanceAdj 类的实例**************
        self.disAdj = DistanceAdj()
        # 创建两个 FHyperGCN 类的实例，分别用于特征相似性图卷积网络和时间关系图卷积网络
        self.HFSGCN = FHyperGCN(args)
        self.HTRGCN = FHyperGCN(args)
        self.HFSGCN1 = FHyperGCN(args)
        self.HTRGCN1 = FHyperGCN(args)
        self.dropoutGCN = nn.Dropout(args.dropoutGCN)
        self.dropoutFC = nn.Dropout(args.dropout0)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.args=args
    # 前向传播函数，接收音频特征f_a、视觉特征f_v和序列长度seq_len
    def forward(self, f_a, f_v, seq_len):
        # 将视觉特征和音频特征分别通过全连接层进行维度变换
        f_v, f_a = self.fc_v(f_v), self.fc_a(f_a)
        _, _, similarity_matrix = self.mra(f_v, f_a)
        scaled_bias = similarity_matrix * self.mra_bias_scaler
        # f_v 保持不变, f_a 被重排以最佳匹配 f_v
        # 将变换后的视觉特征和音频特征输入到交叉注意力块中，得到输出
        #v_out, a_out = self.cma(f_v,f_a)
        v_out, a_out = self.cma(f_v, f_a, attention_bias=scaled_bias)
        v_out1=v_out
        a_out1=a_out
        #print('----------------------------')
        #print('----------------------------')
        #print("v_out shape:", v_out.shape)
        v_out1 = self.fc_ew(v_out1)
        v_out1 = self.dropoutFC(v_out1)
        a_out1 = self.fc_aew(a_out1)
        a_out1 = self.dropoutFC(a_out1)
        #print('222222222222222222222222222222')
        #print("v_out shape:", v_out.shape)
        #/////////////////////////////////////////////
        disadj = self.disAdj(a_out1.shape[0], a_out1.shape[1], self.args).to(a_out1.device)
        # 调用 expm 方法，对输入进行指数映射操作
        proj_x = self.expm(a_out1)
        #print('----------------------------')
        #print("proj_x shape:", proj_x.shape)
        # 调用 adj 方法，计算邻接矩阵
        adj = self.adj(proj_x, seq_len)
        #print('----------------------------')
        #print("adj shape:", adj.shape)
        # 调用 HFSGCN 的 encode 方法进行特征提取，并使用 LeakyReLU 激活函数
        # a_out11 = self.relu(self.HFSGCN.encode(proj_x, adj))
        # # 对提取的特征应用 Dropout 操作
        # a_out11 = self.dropoutGCN(a_out11)
        # 调用 HTRGCN 的 encode 方法进行特征提取，并使用 LeakyReLU 激活函数
        a_out12 = self.relu(self.HTRGCN.encode(proj_x, disadj))
        # 对提取的特征应用 Dropout 操作
        a_out12 = self.dropoutGCN(a_out12)
        # 将交叉注意力块的输出输入到Att_MMIL模块中，得到不同的logits
        # 将两个特征在最后一个维度上拼接起来
        a_out1 = a_out12
        a_out1 = self.fc_aew2(a_out1)
        a_out1 = self.dropoutFC(a_out1)
        # /////////////////////////////////////////////
        disadj = self.disAdj(v_out1.shape[0], v_out1.shape[1], self.args).to(v_out1.device)
        # 调用 expm 方法，对输入进行指数映射操作
        proj_x = self.expm(v_out1)
        # print('----------------------------')
        # print("proj_x shape:", proj_x.shape)
        # 调用 adj 方法，计算邻接矩阵
        adj = self.adj(proj_x, seq_len)
        # print('----------------------------')
        # print("adj shape:", adj.shape)
        # 调用 HFSGCN 的 encode 方法进行特征提取，并使用 LeakyReLU 激活函数
        # v_out21 = self.relu(self.HFSGCN1.encode(proj_x, adj))
        # # print('123456789')
        # # 对提取的特征应用 Dropout 操作
        # v_out21 = self.dropoutGCN(v_out21)
        # 调用 HTRGCN 的 encode 方法进行特征提取，并使用 LeakyReLU 激活函数
        v_out22 = self.relu(self.HTRGCN1.encode(proj_x, disadj))
        # 对提取的特征应用 Dropout 操作
        v_out22 = self.dropoutGCN(v_out22)
        # 将交叉注意力块的输出输入到Att_MMIL模块中，得到不同的logits
        # 将两个特征在最后一个维度上拼接起来
        v_out1 = v_out22
        v_out1 = self.fc_ew2(v_out1)
        v_out1 = self.dropoutFC(v_out1)
        mmil_logits, audio_logits, visual_logits, av_logits = self.att_mmil(a_out1, v_out1, seq_len)
        # 返回各种logits以及交叉注意力块的输出
        return mmil_logits, audio_logits, visual_logits, av_logits, v_out1, a_out1
    def expm(self, x):
        # 如果使用的流形是 Lorentz 或 Hyperboloid
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            # 创建一个与 x 形状相同的全零张量
            o = torch.zeros_like(x)
            # 在 x 的最后一个维度上拼接一个全零列
            x = torch.cat([o[:, :, 0:1], x], dim=-1)
            # 如果使用的流形是 Lorentz，则对 x 进行指数映射操作
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
            return x
        else:
            return x
    def adj(self, x, seq_len):
        # 创建一个 Softmax 激活函数层，用于在第 1 维上进行归一化
        soft = nn.Softmax(1)
        # 调用 lorentz_similarity 方法，计算输入 x 之间的 Lorentz 相似度
        x2 = self.lorentz_similarity(x, x, self.manifold.k)
        # 对相似度矩阵取负指数
        x2 = torch.exp(-x2)
        # 创建一个与 x2 形状相同的全零张量
        output = torch.zeros_like(x2)
        # 如果序列长度为 None
        if seq_len is None:
            # 遍历每个样本
            for i in range(x.shape[0]):
                # 提取当前样本的相似度矩阵
                tmp = x2[i]
                adj2 = tmp
                # 对相似度矩阵进行阈值处理，小于 0.8 的值置为 0
                adj2 = F.threshold(adj2, 0.8, 0)
                # 对阈值处理后的矩阵进行 Softmax 归一化
                adj2 = soft(adj2)
                # 将归一化后的矩阵赋值给输出矩阵的对应位置
                output[i] = adj2
        else:
            # 遍历每个样本的序列长度
            for i in range(len(seq_len)):
                # 提取当前样本的有效相似度矩阵
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                # 对相似度矩阵进行阈值处理，小于 0.8 的值置为 0
                adj2 = F.threshold(adj2, 0.8, 0)
                # 对阈值处理后的矩阵进行 Softmax 归一化
                adj2 = soft(adj2)
                # 将归一化后的矩阵赋值给输出矩阵的对应位置
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def clas(self, logits, seq_len):
        # 对 logits 进行压缩操作，去除维度为 1 的维度
        logits = logits.squeeze()
        # 创建一个空的张量，用于存储实例级别的 logits
        instance_logits = torch.zeros(0).to(logits.device)
        # 遍历每个样本的 logits
        for i in range(logits.shape[0]):
            # 如果序列长度为 None
            if seq_len is None:
                # 计算当前样本的 logits 的平均值，并调整形状为 (1,)
                tmp = torch.mean(logits[i]).view(1)
            else:
                # 从当前样本的有效 logits 中选取前 k 个最大值，k 为序列长度除以 16 向下取整再加 1
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='floor') + 1),
                                    largest=True)
                # 计算选取的最大值的平均值，并调整形状为 (1,)
                tmp = torch.mean(tmp).view(1)
            # 将当前样本的平均值拼接起来
            instance_logits = torch.cat((instance_logits, tmp))
        # 对实例级别的 logits 应用 Sigmoid 激活函数
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def lorentz_similarity(self, x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
        '''
        d = <x, y>   lorentz metric
        '''
        # 定义不同数据类型的最小阈值
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}
        # 创建一个数组，第一个元素为 -1，其余元素为 1
        idx = np.concatenate((np.array([-1]), np.ones(x.shape[-1] - 1)))
        # 将 idx 数组转换为对角矩阵，并转换为 PyTorch 张量，然后移动到指定设备上
        diag = torch.from_numpy(np.diag(idx).astype(np.float32)).to(x.device)
        # 将输入 x 与对角矩阵相乘
        temp = x @ diag
        # 计算 temp 与 y 的转置矩阵的乘积，并取负
        xy_inner = -(temp @ y.transpose(-1, -2))
        # 对 xy_inner 进行阈值处理，小于 1 的值置为 1
        xy_inner_ = F.threshold(xy_inner, 1, 1)
        # 计算 k 的平方根
        sqrt_k = k**0.5
        # 计算距离，公式为 sqrt_k 乘以 arccosh(xy_inner_ / k)
        dist = sqrt_k * self.arccosh(xy_inner_ / k)
        # 对距离进行裁剪，确保其在最小阈值和 200 之间
        dist = torch.clamp(dist, min=self.eps[x.dtype], max=200)
        return dist

    def arccosh(self, x):
        """
        Element-wise arcosh operation.
        Parameters
        ---
        x : torch.Tensor[]
        Returns
        ---
        torch.Tensor[]
            arcosh result.
        """
        # 计算元素级别的反双曲余弦值
        return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))
# 定义Att_MMIL类，继承自nn.Module，用于多模态交互
class Att_MMIL(nn.Module):
    # 初始化函数，接收输入维度input_dim和类别数num_classes
    def __init__(self, input_dim, num_classes,args):
        # 调用父类的初始化方法
        super(Att_MMIL, self).__init__()
        # 定义一个全连接层，将输入维度映射到类别数
        self.fc = nn.Linear(input_dim, num_classes)
    # 分类函数，用于对logits进行处理
    def clas(self, logits, seq_len):
        # 去除logits的单维度
        logits = logits.squeeze()
        # 初始化一个空的张量，用于存储实例级别的logits
        instance_logits = torch.zeros(0, device=logits.device)
        # 遍历每个样本
        for i in range(logits.shape[0]):
            # 如果序列长度seq_len为None
            if seq_len is None:
                # 计算当前样本logits的均值，并调整形状
                tmp = torch.mean(logits[i]).view(1)
            else:
                # 从当前样本的logits中选取前seq_len[i]个元素，并取前seq_len[i] // 16 + 1个最大值
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                # 计算这些最大值的均值，并调整形状
                tmp = torch.mean(tmp).view(1)
            # 将当前样本的处理结果拼接到instance_logits中
            instance_logits = torch.cat((instance_logits, tmp))
        # 对instance_logits应用sigmoid函数
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    # 前向传播函数，接收音频输出a_out、视觉输出v_out和序列长度seq_len
    def forward(self, a_out, v_out, seq_len):
        # 在新的维度上拼接音频和视觉输出
        x = torch.cat([a_out.unsqueeze(-2), v_out.unsqueeze(-2)], dim=-2)
        # 将拼接后的结果通过全连接层
        frame_prob = self.fc(x)
        # 在维度2上求和得到av_logits
        av_logits = frame_prob.sum(dim=2)
        # 对frame_prob的音频部分应用sigmoid函数
        a_logits = torch.sigmoid(frame_prob[:, :, 0, :])
        # 对frame_prob的视觉部分应用sigmoid函数
        v_logits = torch.sigmoid(frame_prob[:, :, 1, :])
        # 调用clas函数对av_logits进行处理
        mmil_logits = self.clas(av_logits, seq_len)
        # 返回各种logits
        return mmil_logits, a_logits, v_logits, av_logits
# 定义Single_Model类，继承自nn.Module，用于构建单模态的神经网络模型
class Single_Model(nn.Module):
    # 初始化函数，接收参数args
    def __init__(self, args):
        # 调用父类的初始化方法
        super(Single_Model, self).__init__()
        # 定义一个深拷贝函数，用于后续复制某些模块
        c = copy.deepcopy
        # 从参数args中获取dropout比例
        dropout = args.dropout
        # 从参数args中获取多头注意力的头数
        nhead = args.nhead
        # 从参数args中获取隐藏层的维度
        hid_dim = args.hid_dim
        # 从参数args中获取前馈网络的维度
        ffn_dim = args.ffn_dim
        # 从参数args中获取视觉特征的维度
        n_dim = args.v_feature_size

        self.manifold = getattr(manifolds, args.manifold)()
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1
        # 初始化一个多头注意力模块
        self.multiheadattn = MultiHeadAttention(nhead, hid_dim)
        # 初始化一个位置前馈网络模块
        self.feedforward = PositionwiseFeedForward(hid_dim, ffn_dim)
        # 定义一个全连接层，将n_dim维的特征映射到hid_dim维
        self.fc_v = nn.Linear(n_dim, hid_dim)
        # 初始化一个自注意力块，其中包含一个Transformer层
        self.cma = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), c(self.feedforward), dropout))
        # 定义一个全连接层，将hid_dim维的特征映射到类别数
        self.fc = nn.Linear(hid_dim, args.num_classes)
        #+++++++++++++++++++++++++++++++++++++++

        self.fc_ew = nn.Linear(128, 256)
        self.fc_ew2 = nn.Linear(32, 128)
        # 创建 DistanceAdj 类的实例**************
        self.disAdj = DistanceAdj()
        # 创建第一个一维卷积层，输入通道数为 1024，输出通道数为 512，卷积核大小为 1，填充为 0
        #self.conv1d1 = nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
        # 创建第二个一维卷积层，输入通道数为 512，输出通道数为 128，卷积核大小为 1，填充为 0
        #self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1, padding=0)
        # 创建两个 FHyperGCN 类的实例，分别用于特征相似性图卷积网络和时间关系图卷积网络
        self.HFSGCN1 = FHyperGCN(args)
        self.HTRGCN1 = FHyperGCN(args)
        self.dropoutGCN = nn.Dropout(args.dropoutGCN)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.args = copy.deepcopy(args)  # 深拷贝参数
        self.dropoutFC = nn.Dropout(args.dropout0)
    # 分类函数，用于对logits进行处理
    def clas(self, logits, seq_len):
        # 去除logits的单维度
        logits = logits.squeeze()
        # 初始化一个空的张量，用于存储实例级别的logits
        instance_logits = torch.zeros(0, device=logits.device)
        # 遍历每个样本
        for i in range(logits.shape[0]):
            # 从当前样本的logits中选取前seq_len[i]个元素，并取前seq_len[i] // 16 + 1个最大值
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            # 计算这些最大值的均值，并调整形状
            tmp = torch.mean(tmp).view(1)
            # 将当前样本的处理结果拼接到instance_logits中
            instance_logits = torch.cat((instance_logits, tmp))
        # 对instance_logits应用sigmoid函数
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    # 前向传播函数，接收特征f和序列长度seq_len
    def forward(self, f, seq_len):
        # 将特征f通过全连接层进行维度变换
        f = self.fc_v(f)
        # 将变换后的特征输入到自注意力块中，得到输出
        sa = self.cma(f)
        sa = self.fc_ew(sa)
        sa = self.dropoutFC(sa)
        disadj = self.disAdj(sa.shape[0], sa.shape[1], self.args).to(sa.device)
        # 调用 expm 方法，对输入进行指数映射操作
        proj_x = self.expm(sa)
        # 调用 adj 方法，计算邻接矩阵
        adj = self.adj(proj_x, seq_len)
        # 调用 HFSGCN 的 encode 方法进行特征提取，并使用 LeakyReLU 激活函数
        # sa1 = self.relu(self.HFSGCN1.encode(proj_x, adj))
        # # 对提取的特征应用 Dropout 操作
        # sa1 = self.dropoutGCN(sa1)
        # 调用 HTRGCN 的 encode 方法进行特征提取，并使用 LeakyReLU 激活函数
        sa2 = self.relu(self.HTRGCN1.encode(proj_x, disadj))
        # 对提取的特征应用 Dropout 操作
        sa2 = self.dropoutGCN(sa2)
        # 将两个特征在最后一个维度上拼接起来
        sa = sa2
        sa = self.fc_ew2(sa)
        sa = self.dropoutFC(sa)
        # 将自注意力块的输出通过全连接层
        out = self.fc(sa)
        # 如果序列长度seq_len不为None
        if seq_len is not None:
            # 调用clas函数对out进行处理
            out = self.clas(out, seq_len)
        # 返回处理后的结果
        return out
    def expm(self, x):
        # 如果使用的流形是 Lorentz 或 Hyperboloid
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            # 创建一个与 x 形状相同的全零张量
            o = torch.zeros_like(x)
            # 在 x 的最后一个维度上拼接一个全零列
            x = torch.cat([o[:, :, 0:1], x], dim=-1)
            # 如果使用的流形是 Lorentz，则对 x 进行指数映射操作
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
            return x
        else:
            return x
    def adj(self, x, seq_len):
        # 创建一个 Softmax 激活函数层，用于在第 1 维上进行归一化
        soft = nn.Softmax(1)
        # 调用 lorentz_similarity 方法，计算输入 x 之间的 Lorentz 相似度
        x2 = self.lorentz_similarity(x, x, self.manifold.k)
        # 对相似度矩阵取负指数
        x2 = torch.exp(-x2)
        # 创建一个与 x2 形状相同的全零张量
        output = torch.zeros_like(x2)
        # 如果序列长度为 None
        if seq_len is None:
            # 遍历每个样本
            for i in range(x.shape[0]):
                # 提取当前样本的相似度矩阵
                tmp = x2[i]
                adj2 = tmp
                # 对相似度矩阵进行阈值处理，小于 0.8 的值置为 0
                adj2 = F.threshold(adj2, 0.8, 0)
                # 对阈值处理后的矩阵进行 Softmax 归一化
                adj2 = soft(adj2)
                # 将归一化后的矩阵赋值给输出矩阵的对应位置
                output[i] = adj2
        else:
            # 遍历每个样本的序列长度
            for i in range(len(seq_len)):
                # 提取当前样本的有效相似度矩阵
                tmp = x2[i, :seq_len[i], :seq_len[i]]
                adj2 = tmp
                # 对相似度矩阵进行阈值处理，小于 0.8 的值置为 0
                adj2 = F.threshold(adj2, 0.8, 0)
                # 对阈值处理后的矩阵进行 Softmax 归一化
                adj2 = soft(adj2)
                # 将归一化后的矩阵赋值给输出矩阵的对应位置
                output[i, :seq_len[i], :seq_len[i]] = adj2
        return output

    def clas(self, logits, seq_len):
        # 对 logits 进行压缩操作，去除维度为 1 的维度
        logits = logits.squeeze()
        # 创建一个空的张量，用于存储实例级别的 logits
        instance_logits = torch.zeros(0).to(logits.device)
        # 遍历每个样本的 logits
        for i in range(logits.shape[0]):
            # 如果序列长度为 None
            if seq_len is None:
                # 计算当前样本的 logits 的平均值，并调整形状为 (1,)
                tmp = torch.mean(logits[i]).view(1)
            else:
                # 从当前样本的有效 logits 中选取前 k 个最大值，k 为序列长度除以 16 向下取整再加 1
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(torch.div(seq_len[i], 16, rounding_mode='floor') + 1),
                                    largest=True)
                # 计算选取的最大值的平均值，并调整形状为 (1,)
                tmp = torch.mean(tmp).view(1)
            # 将当前样本的平均值拼接起来
            instance_logits = torch.cat((instance_logits, tmp))
        # 对实例级别的 logits 应用 Sigmoid 激活函数
        instance_logits = torch.sigmoid(instance_logits)
        return instance_logits

    def lorentz_similarity(self, x: torch.Tensor, y: torch.Tensor, k) -> torch.Tensor:
        '''
        d = <x, y>   lorentz metric
        '''
        # 定义不同数据类型的最小阈值
        self.eps = {torch.float32: 1e-6, torch.float64: 1e-8}
        # 创建一个数组，第一个元素为 -1，其余元素为 1
        idx = np.concatenate((np.array([-1]), np.ones(x.shape[-1] - 1)))
        # 将 idx 数组转换为对角矩阵，并转换为 PyTorch 张量，然后移动到指定设备上
        diag = torch.from_numpy(np.diag(idx).astype(np.float32)).to(x.device)
        # 将输入 x 与对角矩阵相乘
        temp = x @ diag
        # 计算 temp 与 y 的转置矩阵的乘积，并取负
        xy_inner = -(temp @ y.transpose(-1, -2))
        # 对 xy_inner 进行阈值处理，小于 1 的值置为 1
        xy_inner_ = F.threshold(xy_inner, 1, 1)
        # 计算 k 的平方根
        sqrt_k = k**0.5
        # 计算距离，公式为 sqrt_k 乘以 arccosh(xy_inner_ / k)
        dist = sqrt_k * self.arccosh(xy_inner_ / k)
        # 对距离进行裁剪，确保其在最小阈值和 200 之间
        dist = torch.clamp(dist, min=self.eps[x.dtype], max=200)
        return dist

    def arccosh(self, x):
        """
        Element-wise arcosh operation.
        Parameters
        ---
        x : torch.Tensor[]
        Returns
        ---
        torch.Tensor[]
            arcosh result.
        """
        # 计算元素级别的反双曲余弦值
        return torch.log(x + torch.sqrt(torch.pow(x, 2) - 1))