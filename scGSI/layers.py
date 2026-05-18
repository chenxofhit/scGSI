import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv
from torch.distributions import NegativeBinomial


class CrossAttentionFusion(nn.Module):
    """
    使用交叉注意力融合RNA和ATAC的嵌入。
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        # 注意力层：ATAC作为query，RNA作为key/value
        self.cross_attn_a_to_r = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # 注意力层：RNA作为query，ATAC作为key/value
        self.cross_attn_r_to_a = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ffn_rna = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_atac = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

    def forward(self, z_rna, z_atac):
        # z_rna 和 z_atac 需要是 (batch, seq_len, embed_dim) 的形状
        # 对于单细胞数据, seq_len 是细胞数量，batch是1
        z_rna_b = z_rna.unsqueeze(0)  # (1, num_rna_cells, dim)
        z_atac_b = z_atac.unsqueeze(0)  # (1, num_atac_cells, dim)

        # ATAC 从 RNA 查询信息
        attn_output_a, _ = self.cross_attn_a_to_r(query=z_atac_b, key=z_rna_b, value=z_rna_b)
        z_atac_fused = self.norm1(z_atac + attn_output_a.squeeze(0))
        z_atac_fused = self.norm2(z_atac_fused + self.ffn_atac(z_atac_fused))

        # RNA 从 ATAC 查询信息
        attn_output_r, _ = self.cross_attn_r_to_a(query=z_rna_b, key=z_atac_b, value=z_atac_b)
        z_rna_fused = self.norm3(z_rna + attn_output_r.squeeze(0))
        z_rna_fused = self.norm4(z_rna_fused + self.ffn_rna(z_rna_fused))

        return z_rna_fused, z_atac_fused


class GraphSAGEEncoder(nn.Module):
    """
    输入：基因表达矩阵 (N × G, 连续值) 和 KNN 图 (edge_index)。
    输出：低维潜在表示 Z (N × Z)。
    """

    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.3):
        """
        参数:
            in_dim (int): 输入维度（基因数 G）。
            hidden_dim (int): 隐藏层维度。
            latent_dim (int): 输出维度（潜在表示 Z）。
            num_layers (int): GraphSAGE 层数。
            dropout (float): Dropout 比率，防止过拟合。
        """
        super(GraphSAGEEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 定义 GraphSAGE 层
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, latent_dim))

        # 批归一化
        self.bns = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, x, x_edge_index):
        """
        参数:
            x (torch.Tensor): 输入特征矩阵，形状 (N, G)，RNA 基因表达。
            x_edge_index (torch.Tensor): KNN 图边索引，形状 (2, E)，E 为边数：样本数*k。
        返回:
            z_rna (torch.Tensor): 潜在表示，形状 (N, Z)。
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, x_edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        z_rna = self.convs[-1](x, x_edge_index)  # 最后一层不加激活
        return z_rna


class GATv2Encoder(nn.Module):
    """
    输入：峰值矩阵 (N × P, 二值) 和 KNN 图 (edge_index)。
    输出：低维潜在表示 Z (N × Z)。
    """

    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers=2, num_heads=4, dropout=0.3):
        """
        参数:
            in_dim (int): 输入维度（峰值数 P）。
            hidden_dim (int): 隐藏层维度。
            latent_dim (int): 输出维度（潜在表示 Z）。
            num_layers (int): GATv2 层数。
            heads (int): 注意力头数，增强特征表达。
            dropout (float): Dropout 比率。
        """
        super(GATv2Encoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 定义 GATv2 层
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim // num_heads, heads=num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        self.convs.append(GATv2Conv(hidden_dim, latent_dim, heads=1))  # 最后一层单头

        # # 批归一化
        # self.bns = nn.ModuleList()
        # for _ in range(num_layers - 1):
        #     self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, y, y_edge_index):
        """
        参数:
            y (torch.Tensor): 输入特征矩阵，形状 (N, P)，ATAC 峰值矩阵。
            y_edge_index (torch.Tensor): KNN 图边索引，形状 (2, E)。
        返回:
            z_atac (torch.Tensor): 潜在表示，形状 (N, Z)。
        """
        for i in range(self.num_layers - 1):
            y = self.convs[i](y, y_edge_index)
            # y = self.bns[i](y)
            y = F.elu(y)  # ELU 激活适合稀疏数据
            y = F.dropout(y, p=self.dropout, training=self.training)
        z_atac = self.convs[-1](y, y_edge_index)
        return z_atac


class STEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, latent_dim, num_heads=4, num_layers=2, dropout=0.3, residual=True):
        super(STEncoder, self).__init__()
        self.pre_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            heads = num_heads if i < num_layers - 1 else 1  # 最后一层单头
            concat = i < num_layers - 1
            self.gat_layers.append(GATv2Conv(latent_dim, latent_dim // heads if concat else latent_dim,
                                             heads=heads, concat=concat, dropout=dropout))

        self.post_fc = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.norm = nn.LayerNorm(latent_dim)  # 可加，防 over-smoothing

    def forward(self, x_st, st_edge_index):
        z = self.pre_fc(x_st)  # 先投影到 latent 空间

        resid = z if self.residual else None
        for gat in self.gat_layers:
            z = gat(z, st_edge_index)
            z = F.relu(z)  # 或 elu
            z = self.dropout(z)
            if self.residual:
                z = z + resid
                resid = z
            z = self.norm(z)

        z = self.post_fc(z)
        return z


class MLPDecoder(nn.Module):
    """
    A multilayer perceptron (MLP) decoder for reconstructing high-dimensional features from low-dimensional latent embeddings.
    Designed for single-cell multi-omics data, it supports dropout for regularization and normal initialization for stable training.
    """

    def __init__(self, latent_dim, hidden_dim, feature_dim, dropout=0.3):
        """
        Initializes the MLP decoder.

        Args:
            latent_dim (int): Dimensionality of the input latent embeddings.
            hidden_dim (int): Dimensionality of the hidden layer.
            feature_dim (int): Dimensionality of the output reconstructed features (e.g., gene or peak count).
            dropout (float): Dropout rate to prevent overfitting, particularly suited for sparse data.
        """
        super(MLPDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # First linear layer: expands latent space to hidden dimension
            nn.ReLU(),  # ReLU's activation for non-linearity
            nn.Dropout(dropout),  # Dropout layer to mitigate overfitting and adapt to sparse inputs
            nn.Linear(hidden_dim, feature_dim)  # Second linear layer: projects to original feature space
        )
        # Apply normal initialization to linear layers for stable gradient flow
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0, std=0.1)
                torch.nn.init.normal_(module.bias, mean=0, std=0.1)

    def forward(self, z):
        """
        Forward pass of the decoder.

        Args:
            z (torch.Tensor): Input latent embeddings, shape (N, latent_dim), where N is the number of cells.

        Returns:
            torch.Tensor: Reconstructed features, shape (N, feature_dim).
        """
        x = self.fc(z)
        return x


