import torch
import torch.nn as nn
import torch.nn.functional as F


'''
RVQ（残差向量量化）在PyTorch中的实现

(1) 初始化:
num_codebooks: 码本的数量 K。
codebook_size: 每个码本中向量的数量 M。
embedding_dim: 向量的维度 D。
self.codebooks: 使用 nn.ModuleList 存储 K 个 nn.Embedding 层，每个 nn.Embedding 层代表一个码本。

(2) 前向传播 (forward 方法):
输入张量 x 的形状为 (batch_size, sequence_length, embedding_dim)。
初始化残差 residual 为输入张量 x 的副本。
遍历每个码本：
计算输入张量与当前码本中所有向量的距离。
使用 argmin 找到距离最近的码本向量索引。
使用 nn.Embedding 根据索引获取量化后的向量。
更新残差 residual，即从输入中减去量化后的向量。
最终的量化结果 quantized 为输入张量减去总残差。

(3) 获取量化向量 (get_quantized_vectors 方法):
根据量化索引列表 indices 和码本，重建量化后的向量。
该方法用于在推理阶段，根据保存的量化索引生成量化后的音频信号。

'''

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        """
        初始化RVQ量化器。

        Args:
            num_codebooks (int): 码本的数量K。
            codebook_size (int): 每个码本中向量数量M。
            embedding_dim (int): 向量维度D。
        """
        super(ResidualVectorQuantizer, self).__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

        # 初始化K个码本，每个码本包含M个D维向量
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, embedding_dim) for _ in range(num_codebooks)
        ])

        # 初始化码本向量为均匀分布
        self._init_codebooks()

    def _init_codebooks(self):
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def forward(self, x):
        """
        前向传播，进行量化。

        Args:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length, embedding_dim)。

        Returns:
            quantized (torch.Tensor): 量化后的张量，形状与x相同。
            indices (list of torch.Tensor): 每个码本的量化索引，列表长度为K，每个元素形状为 (batch_size, sequence_length)。
        """
        batch_size, sequence_length, _ = x.shape
        residual = x.clone()
        indices = []

        for k, codebook in enumerate(self.codebooks):
            # 计算与码本中每个向量的距离
            # (batch_size, sequence_length, codebook_size)
            distances = torch.norm(residual.unsqueeze(-2) - codebook.weight, dim=-1)

            # 找到最近的码本索引
            index = torch.argmin(distances, dim=-1)  # (batch_size, sequence_length)
            indices.append(index)

            # 获取对应的码本向量
            # (batch_size, sequence_length, embedding_dim)
            quantized = F.embedding(index, codebook.weight)

            # 更新残差
            residual = residual - quantized

        # 最终的量化结果为所有码本向量的和
        quantized = x - residual

        return quantized, indices

    def get_quantized_vectors(self, indices):
        """
        根据量化索引获取量化后的向量。

        Args:
            indices (list of torch.Tensor): 每个码本的量化索引。

        Returns:
            quantized (torch.Tensor): 量化后的向量，形状为 (batch_size, sequence_length, embedding_dim)。
        """
        quantized = 0
        for k, index in enumerate(indices):
            quantized += F.embedding(index, self.codebooks[k].weight)
        return quantized
    

# 参数设置
num_codebooks = 4
codebook_size = 2048
embedding_dim = 64
batch_size = 16
sequence_length = 100

# 随机输入张量
x = torch.randn(batch_size, sequence_length, embedding_dim)

# 初始化RVQ量化器
rvq = ResidualVectorQuantizer(num_codebooks, codebook_size, embedding_dim)

# 前向传播
quantized, indices = rvq(x)

# 查看输出形状
print("Quantized shape:", quantized.shape)  # (batch_size, sequence_length, embedding_dim)
print("Number of indices lists:", len(indices))  # K
print("Indices shape:", indices[0].shape)   # (batch_size, sequence_length)

# 根据量化索引重建量化向量
reconstructed = rvq.get_quantized_vectors(indices)
print("Reconstructed shape:", reconstructed.shape)  # (batch_size, sequence_length, embedding_dim)
