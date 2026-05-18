import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial, Bernoulli


class x_recoLoss(nn.Module):
    """
    重构损失RNA 数据。
    参数:
        pos_weight (float): 正样本权重，平衡稀疏性（默认 1.0）。
        eps (float): 数值稳定性常数（默认 1e-10）。
        l2_lambda (float): L2 正则化权重（默认 1e-4）。
    """

    def __init__(self, pos_weight=1.0, eps=1e-10, l2_lambda=1e-4, use_nb=False):
        super(x_recoLoss, self).__init__()
        self.pos_weight = pos_weight
        self.eps = eps
        self.l2_lambda = l2_lambda
        self.use_nb = use_nb  # 是否使用负二项分布

    def forward(self, x, reco_x, model):
        """
        参数:
            x (torch.Tensor): 输入到编码器的 RNA 数据，形状 (batch_size, gene_dim)。
            mu (torch.Tensor): 预测均值，形状 (batch_size, gene_dim)。
            theta (torch.Tensor): 预测离散度，形状 (batch_size, gene_dim)。
            model (nn.Module, optional): 用于 L2 正则化的编码器。

        返回:
            loss (torch.Tensor): 负对数似然损失 + L2 正则化。
        """

        # MSE 损失，适用于归一化数据，x为负值或非整数
        reco_r_loss = F.mse_loss(reco_x, x, reduction='mean') * self.pos_weight

        l2_loss = 0.0
        if model is not None:
            l2_loss = sum(p.pow(2).sum() for p in model.parameters()) * self.l2_lambda

        return reco_r_loss + l2_loss


class y_recoLoss(nn.Module):
    """
    重构损失ATAC 数据
    参数:
        pos_weight (float): 正样本权重，平衡稀疏性（默认 1.0）。
        eps (float): 数值稳定性常数（默认 1e-10）。
        l2_lambda (float): L2 正则化权重（默认 1e-4）。
    """

    def __init__(self, pos_weight=1.0, eps=1e-10, l2_lambda=1e-4, use_binary=False):
        super(y_recoLoss, self).__init__()
        self.pos_weight = pos_weight
        self.eps = eps
        self.l2_lambda = l2_lambda
        self.use_binary = use_binary

    def forward(self, y, reco_y, model):
        """
        参数:
            y (torch.Tensor): 输入到编码器的 ATAC 数据，形状 (batch_size, peak_dim)。
            p (torch.Tensor): 预测概率，形状 (batch_size, peak_dim)。
            model (nn.Module, optional): 用于 L2 正则化的编码器。

        返回:
            loss (torch.Tensor): 加权二元交叉熵损失 + L2 正则化。
        """
        if self.use_binary:
            if not torch.all((y == 0) | (y == 1)):
                raise ValueError("ATAC data must be binary (0 or 1) for binary_cross_entropy")
            reco_a_loss = F.binary_cross_entropy(
                reco_y, y, reduction='mean'
            )
        else:
            reco_a_loss = F.mse_loss(reco_y, y, reduction='mean') * self.pos_weight
        l2_loss = 0.0
        if model is not None:
            l2_loss = sum(p.pow(2).sum() for p in model.parameters()) * self.l2_lambda

        return reco_a_loss + l2_loss


class AdaptiveFeatureLinkedCosineLoss(nn.Module):
    """
    Enhanced Cosine Similarity Alignment Loss with Feature Links and Adaptive Temperature.
    - Feature Links: Weighted by biological correspondence (e.g., gene-peak matrix).
    - Adaptive Temperature: Learnable scaling based on embedding entropy.

    Parameters:
        temperature_init (float): Initial temperature (default: 0.1, from benchmarks).
        link_matrix (torch.Tensor, optional): Pre-computed feature link matrix (e.g., gene-peak overlaps), shape (latent_dim_rna, latent_dim_atac). If None, uniform weights.
        learn_temp (bool): Enable adaptive temperature (default: True).
    """

    def __init__(self, temperature_init=0.1, link_matrix=None, learn_temp=True, *args, **kwargs):
        super().__init__()
        self.temperature_init = temperature_init
        self.link_matrix = link_matrix  # Optional: (latent_dim_rna, latent_dim_atac)
        if link_matrix is not None:
            self.link_matrix = F.normalize(link_matrix.float(), dim=-1)  # Normalize for weights
        self.learn_temp = learn_temp
        if learn_temp:
            self.temp_param = nn.Parameter(torch.tensor(temperature_init))  # Learnable temp

    def forward(self, z_rna, z_atac):
        """
        Parameters:
            z_rna (torch.Tensor): RNA embeddings, shape (batch_size, latent_dim).
            z_atac (torch.Tensor): ATAC embeddings, shape (batch_size, latent_dim).

        Returns:
            loss (torch.Tensor): Enhanced alignment loss.
        """
        batch_size, latent_dim = z_rna.shape
        assert z_atac.shape == z_rna.shape, "Embeddings must have same shape"

        # Step 1: Normalize embeddings to unit vectors
        z_rna_norm = F.normalize(z_rna, p=2, dim=-1)
        z_atac_norm = F.normalize(z_atac, p=2, dim=-1)

        # Step 2: Compute base cosine similarity (paired: diagonal)
        cos_sim = torch.sum(z_rna_norm * z_atac_norm, dim=-1)  # Shape: (batch_size,)

        # Step 3: Feature Link Weighting (if provided)
        if self.link_matrix is not None:
            # Expand link_matrix to batch: (batch, latent_rna, latent_atac) -> weighted sum
            # Assume latent_dim same; for diff dims, adjust broadcasting
            link_weights = self.link_matrix.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, d, d)
            weighted_cos = torch.sum(link_weights * (z_rna_norm.unsqueeze(-1) * z_atac_norm.unsqueeze(1)),
                                     dim=(1, 2))  # (batch,)
            cos_sim = weighted_cos  # Apply weights
        else:
            # Uniform if no links
            pass

        # Step 4: Adaptive Temperature Scaling
        if self.learn_temp:
            # Adaptive: tau = sigmoid(temp_param) * init + (1 - sigmoid) * entropy-based adjust
            entropy_rna = -torch.sum(z_rna_norm * torch.log(z_rna_norm + 1e-8), dim=-1).mean()  # Avg embedding entropy
            entropy_atac = -torch.sum(z_atac_norm * torch.log(z_atac_norm + 1e-8), dim=-1).mean()
            avg_entropy = (entropy_rna + entropy_atac) / 2
            adaptive_scale = torch.sigmoid(self.temp_param) * self.temperature_init + (
                        1 - torch.sigmoid(self.temp_param)) * avg_entropy
            tau = adaptive_scale.clamp(min=0.01, max=1.0)  # Clamp for stability
        else:
            tau = self.temperature_init

        # Scaled cosine
        scaled_cos = cos_sim / tau

        # Step 5: Negative mean loss (maximize similarity)
        loss = -torch.mean(scaled_cos)

        return loss


class ContrastiveLoss(nn.Module):
    """对比学习损失函数"""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, rna_features, atac_features, labels=None):
        """
        Args:
            rna_features: [batch_size, hidden_size]
            atac_features: [batch_size, hidden_size]
            labels: [batch_size] - 用于硬负样本挖掘
        """
        # 归一化特征
        rna_features = F.normalize(rna_features, dim=1)
        atac_features = F.normalize(atac_features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(rna_features, atac_features.T) / self.temperature

        # 对角线元素是正样本对
        positive_pairs = torch.diag(similarity_matrix)

        # 计算InfoNCE损失
        # 对于每个RNA特征，ATAC特征作为正样本，其他所有ATAC特征作为负样本
        rna_to_atac_loss = -torch.log(
            torch.exp(positive_pairs) /
            torch.sum(torch.exp(similarity_matrix), dim=1)
        ).mean()

        # 对于每个ATAC特征，RNA特征作为正样本，其他所有RNA特征作为负样本
        atac_to_rna_loss = -torch.log(
            torch.exp(positive_pairs) /
            torch.sum(torch.exp(similarity_matrix.T), dim=1)
        ).mean()

        # 如果提供了标签，可以进行硬负样本挖掘
        if labels is not None:
            # 创建标签掩码，相同标签的样本作为硬负样本
            label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            # 移除对角线（自己）
            label_mask = label_mask - torch.eye(label_mask.size(0), device=label_mask.device)

            # 硬负样本损失
            hard_negative_loss = torch.sum(
                label_mask * torch.exp(similarity_matrix), dim=1
            ) / (torch.sum(label_mask, dim=1) + 1e-8)
            hard_negative_loss = torch.log(hard_negative_loss).mean()

            return (rna_to_atac_loss + atac_to_rna_loss) / 2 + 0.1 * hard_negative_loss

        return (rna_to_atac_loss + atac_to_rna_loss) / 2


class CosineAlignmentLoss(nn.Module):
    """
    低 使正对（同一细胞模态）相似度更突出（强调对齐），高 使分布更平滑（减少噪声敏感）
    Cosine Similarity Alignment Loss, adapted from scPairing for DaOT.
    Encourages similarity between RNA and ATAC embeddings before fusion.
    Computes negative mean cosine similarity on normalized embeddings.
    应负值趋近 -1，表示高相似
    Parameters:
        temperature (float, optional): Scaling factor for cosine similarity (default: 1.0, no scaling).
    """
    def __init__(self, temperature=0.1):
        super(CosineAlignmentLoss, self).__init__()
        self.temperature = temperature
        # Learnable log-temperature for adaptive scaling (ensures temperature > 0)
        # self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))

    def forward(self, z_rna, z_atac):
        """
        Parameters:
            z_rna (torch.Tensor): RNA embeddings from GraphSAGEEncoder, shape (batch_size, latent_dim).
            z_atac (torch.Tensor): ATAC embeddings from GATv2Encoder, shape (batch_size, latent_dim).
        
        Returns:
            loss (torch.Tensor): Negative mean cosine similarity loss.
        """

        # Normalize embeddings to unit vectors (L2 norm)
        z_rna_norm = F.normalize(z_rna, p=2, dim=-1)
        z_atac_norm = F.normalize(z_atac, p=2, dim=-1)

        # Compute cosine similarity (dot product after normalization)
        # temperature = torch.exp(self.log_temperature)
        # For paired cells (diagonal of similarity matrix)
        cos_sim = torch.sum(z_rna_norm * z_atac_norm, dim=-1) / self.temperature
        
        # Negative mean: minimize loss to maximize similarity
        loss = -torch.mean(cos_sim)
        
        return loss


class RBF(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = nn.Parameter(torch.Tensor((mul_factor ** (torch.arange(n_kernels) - n_kernels // 2))), requires_grad=False)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] /
                         (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def Total_Loss(reco_r_loss, reco_a_loss,
               cos_loss,
               # gw_loss,
               contra_loss,
               nb_model, ber_model, epoch=None,
               nb_weight=0.5, ber_weight=0.7,
               cos_weight=0.7,
               gw_weight=0.05, contra_weight=0.8,
               weight_decay=0.95, max_norm=1.0):
    """
    计算联合损失，组合重构、图结构、GW 和对比损失。

    参数:
        nb_loss (NBReconstructionLoss): RNA 重构损失。
        bernoulli_loss (BernoulliReconstructionLoss): ATAC 重构损失。
        graph_loss (GraphStructureLoss): 图结构损失。
        gw_loss (GromovWassersteinLoss): GW 损失。
        contra_loss (ContrastiveLoss): 对比损失。
        nb_model (nn.Module, optional): NBDecoder 模型。
        bernoulli_model (nn.Module, optional): BernoulliDecoder 模型。
        epoch (int): 当前训练轮数，用于动态权重。
        gw_weight (float): 初始 GW 损失权重（默认 0.1）。
        graph_weight (float): 初始图结构损失权重（默认 0.1）。
        contra_weight (float): 初始对比损失权重（默认 0.1）。
        weight_decay (float): 权重衰减因子（默认 0.9，每 10 个 epoch 衰减）。
        max_norm (float): 梯度裁剪最大范数（默认 1.0）。

    返回:
        total_loss (torch.Tensor): 联合损失。
        loss_dict (dict): 各损失分量的值。
    """
    # 动态调整权重
    decay_factor = weight_decay ** (epoch // 10)
    nb_weight = nb_weight
    ber_weight = ber_weight
    # gw_weight = gw_weight * decay_factor
    cos_weight = cos_weight * decay_factor
    contra_weight = contra_weight * decay_factor

    # 联合损失
    total_loss = (
            nb_weight * reco_r_loss +
            ber_weight * reco_a_loss +
            cos_weight * cos_loss +
            # gw_weight * gw_loss +
            contra_weight * contra_loss
    )

    # 梯度裁剪
    if nb_model is not None and ber_model is not None:
        torch.nn.utils.clip_grad_norm_(
            list(nb_model.parameters()) + list(ber_model.parameters()),
            max_norm=max_norm
        )

    # 记录损失分量
    loss_dict = {
        'epoch': epoch,
        'reco_r_loss': reco_r_loss.item(),
        'reco_a_loss': reco_a_loss.item(),
        # 'gw_loss': gw_loss.item(),
        'cos_loss': cos_loss.item(),
        'contra_loss': contra_loss.item(),
        'nb_weight': nb_weight,
        'ber_weight': ber_weight,
        'cos_weight': cos_weight,
        # 'gw_weight': gw_weight,
        'contra_weight': contra_weight,
    }

    return total_loss, loss_dict
