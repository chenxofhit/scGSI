import os
import pandas as pd
import torch
import numpy as np
from torch import optim, nn
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from scGSI.evaluate import FOSCTTM, evaluate_alignment
from scGSI.layers import (
    GraphSAGEEncoder,
    GATv2Encoder,
    STEncoder,
    MLPDecoder,
    CrossAttentionFusion
)
from scGSI.losses import (x_recoLoss,
                           y_recoLoss,
                           ContrastiveLoss,
                           CosineAlignmentLoss,
                           MMDLoss,
                           Total_Loss)
from scGSI.utils import (get_marginals,
                          Construct_Graph_and_intra_distances,
                          init_random_seed)
from scGSI.preprocess import normalize_data


class scGSi(nn.Module):

    def __init__(self,
                 data,
                 gene_dim,
                 peak_dim,
                 hidden_dim,
                 latent_dim,
                 cell_num,
                 dropout  # to  0.3
                 ):
        super(scGSi, self).__init__()

        self.gene_dim = gene_dim
        self.peak_dim = peak_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.RNA_encoder = GraphSAGEEncoder(
            in_dim=gene_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=2,
            dropout=dropout)
        # self.RNA_decoder = NBDecoder(latent_dim=latent_dim, gene_dim=gene_dim, dropout=dropout)
        self.RNA_decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            feature_dim=gene_dim,
            dropout=dropout)

        self.ATAC_encoder = GATv2Encoder(
            in_dim=peak_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=2,
            num_heads=4,
            dropout=dropout)
        # self.ATAC_decoder = BernoulliDecoder(latent_dim=latent_dim, peak_dim=peak_dim)
        self.ATAC_decoder = MLPDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            feature_dim=peak_dim,
            dropout=dropout)
        # self.ST_encoder = STEncoder(
        #     in_dim=st_gene_dim,
        #     hidden_dim=hidden_dim,
        #     latent_dim=latent_dim,
        #     num_layers=2,
        #     num_heads=4,
        #     dropout=dropout)
        # self.ST_decoder = MLPDecoder(
        #     latent_dim=latent_dim,
        #     hidden_dim=hidden_dim,
        #     feature_dim=st_gene_dim,
        #     dropout=dropout)
        # 新地融合模块
        self.fusion_module = CrossAttentionFusion(latent_dim, dropout=dropout)

        self.graph = Construct_Graph_and_intra_distances(data,
                                                         cell_num=cell_num, device=data[0].device)
        self.RNA_Loss = x_recoLoss()
        self.ATAC_Loss = y_recoLoss()
        self.Cosine_align_loss = CosineAlignmentLoss()
        self.ContrastiveLoss = ContrastiveLoss()
        #self.MMDLoss = MMDLoss()

    def forward(self, rna, rna_edge_index, atac, atac_edge_index, epoch, marginals_mode, graph_mode):
        assert rna.shape[1] == self.gene_dim, \
            f"RNA data dim {rna.shape[1]} does not match gene_dim {self.gene_dim}"
        assert atac.shape[1] == self.peak_dim, \
            f"ATAC data dim {atac.shape[1]} does not match peak_dim {self.peak_dim}"

        z_rna = self.RNA_encoder(rna, rna_edge_index)
        z_atac = self.ATAC_encoder(atac, atac_edge_index)

        cos_loss = self.Cosine_align_loss(z_rna, z_atac)

        z_rna_fused, z_atac_fused = self.fusion_module(z_rna, z_atac)

        contra_loss = self.ContrastiveLoss(z_rna_fused, z_atac_fused)

        latent_data = [z_rna_fused, z_atac_fused]

        reco_x = self.RNA_decoder(z_rna_fused)
        reco_y = self.ATAC_decoder(z_atac_fused)
        reco_data = [reco_x, reco_y]
        reco_r_loss = self.RNA_Loss(rna, reco_x, self.RNA_encoder)
        reco_a_loss = self.ATAC_Loss(atac, reco_y, self.ATAC_encoder)
        # mmd_loss = self.MMDLoss((rna, atac)
        total_loss, loss_dict = Total_Loss(
            reco_r_loss,
            reco_a_loss,
            cos_loss,
            contra_loss,
            # mmd_loss
            self.RNA_encoder, self.ATAC_encoder,
            epoch
        )
        # 返回最终地联合嵌入用于下游任务评估
        z_joint = torch.cat([z_rna_fused, z_atac_fused], dim=0)

        return total_loss, loss_dict, z_joint, reco_data


class scGSi_learning(torch.nn.Module):

    def __init__(self, model, x_raw, y_raw, k_neighbors, k_clusters,
                 learning_rate, weight_decay,
                 num_epochs, seed, model_path, clusters_label, device='cuda'):

        super().__init__()
        self.model = model
        self.data = [x_raw, y_raw]
        self.k_neighbors = k_neighbors
        self.k_clusters = k_clusters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.seed = seed
        self.model_path = model_path
        self.clusters_label = clusters_label
        self.device = device
        self.log_dicts = []
        self.cell_num = [np.shape(data)[0] for data in self.data]
        self.epoch_best = 0

    def train_model(self):
        co_embedding = None
        data = self.data
        batch_size = 256

        init_random_seed(self.seed)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.learning_rate, weight_decay=self.weight_decay)
        pearson_history = []
        best_ari = -float('inf')
        best = 0
        foscttm_best, epoch_best = float('inf'), 0
        stop_patience = 50
        # best = foscttm_best

        for epoch in tqdm(range(1, self.num_epochs + 1), desc="Training Progress"):
            optimizer.zero_grad()

            dataset = TensorDataset(data[0], data[1])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            pearson_dict = {'pearson_RNA': 0.0, 'pearson_ATAC': 0.0}
            epoch_loss_dict = {'nb_loss': 0.0, 'ber_loss': 0.0, 'cos_loss': 0.0, 'contra_loss': 0.0}
            total_loss_sum = 0.0
            batch_count = 0

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                batch_data = [batch_x, batch_y]
                batch_cell_num = [batch_x.shape[0], batch_y.shape[0]]

                # 构建批次图
                get_graph = Construct_Graph_and_intra_distances(batch_data, batch_cell_num, self.device)
                get_graph.construct_knn_graph(graph_mode="distance",
                                              k=self.k_neighbors,
                                              metric="minkowski")
                graph_x = get_graph.graphs[0]
                graph_y = get_graph.graphs[1]

                # 前向传播
                total_loss, loss_dict, z_joint, _ = self.model(batch_data[0], graph_x[0],
                                                               batch_data[1], graph_y[0],
                                                               epoch,
                                                               marginals_mode="uniform",
                                                               graph_mode="distance")

                # 累积损失
                total_loss_sum += total_loss.item()
                for key in epoch_loss_dict:
                    epoch_loss_dict[key] += loss_dict.get(key, 0)
                batch_count += 1

                # 反向传播和优化
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5)
                optimizer.step()

            # 计算每个 epoch 的平均损失
            total_loss_avg = total_loss_sum / batch_count
            for key in epoch_loss_dict:
                epoch_loss_dict[key] /= batch_count

            # 当前批次下（使用完整数据集）计算 FOSCTTM 和其他指标进行模型保存
            with torch.no_grad():
                get_graph = Construct_Graph_and_intra_distances(data, self.cell_num, self.device)
                get_graph.construct_knn_graph(graph_mode="distance",
                                              k=self.k_neighbors,
                                              metric="minkowski")
                graph_x = get_graph.graphs[0]
                graph_y = get_graph.graphs[1]

                _, _, z_joint, reco_data = self.model(data[0], graph_x[0],
                                                      data[1], graph_y[0],
                                                      epoch,
                                                      marginals_mode="uniform", graph_mode="distance")

                # 衡量重构数据与原始数据的线性一致性（重构质量）
                def pearson_correlation(x, y):
                    x_flat = x.view(-1).cpu().numpy()
                    y_flat = y.view(-1).cpu().numpy()
                    return np.corrcoef(x_flat, y_flat)[0, 1]

                corr_rna = pearson_correlation(reco_data[0], data[0])
                corr_atac = pearson_correlation(reco_data[1], data[1])
                pearson_history.append((corr_rna + corr_atac) / 2)
                print(f"Pearson 相关性 RNA: {corr_rna:.4f}")
                print(f"Pearson 相关性 ATAC: {corr_atac:.4f}")
                co_embedding = z_joint

                foscttm = FOSCTTM(z_joint[:self.cell_num[0], ], z_joint[self.cell_num[0]:, ])

                metrics = evaluate_alignment(
                    co_embedding[0:self.cell_num[0]],
                    co_embedding[self.cell_num[0]:],
                    self.clusters_label,
                    k_clusters=self.k_clusters,
                    k_neighbors=self.k_neighbors
                )

            # 保存最佳模型
            # if metrics[2] > best:
            if foscttm < foscttm_best:
                foscttm_best, epoch_best = foscttm, epoch
                torch.save(self.model.state_dict(), self.model_path)

            # 早停逻辑
            if epoch - epoch_best >= stop_patience:
                print(
                    f'在第 {epoch} 轮停止训练。最佳 ARI/foscttm: {best:.4f} 在第 {epoch_best} 轮')
                break

            # 每 10 个 epoch 打印一次指标
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.num_epochs}: Total Loss = {total_loss_avg:.4f}, "
                      f"NB Loss = {epoch_loss_dict['nb_loss']:.4f}, "
                      f"Bernoulli Loss = {epoch_loss_dict['ber_loss']:.4f}, "
                      f"Cosine Loss = {epoch_loss_dict['cos_loss']:.4f}, "
                      f"Contrastive Loss = {epoch_loss_dict['contra_loss']:.4f}, ")

            self.log_dicts.append(epoch_loss_dict)  # 更新 loss_dict

            # self.log_dicts.append(loss_dict)  # 更新 loss_dict

        # 最终评估（使用完整数据集）
        with torch.no_grad():
            metrics = evaluate_alignment(
                co_embedding[0:self.cell_num[0]],
                co_embedding[self.cell_num[0]:],
                self.clusters_label,
                k_clusters=self.k_clusters,
                k_neighbors=self.k_neighbors
            )

            # # 衡量重构数据与原始数据的线性一致性（重构质量）
            # def pearson_correlation(x, y):
            #     x_flat = x.view(-1).cpu().numpy()
            #     y_flat = y.view(-1).cpu().numpy()
            #     return np.corrcoef(x_flat, y_flat)[0, 1]
            #
            # reco_x = self.model.RNA_decoder(co_embedding[0:self.cell_num[0]])
            # reco_y = self.model.ATAC_decoder(co_embedding[self.cell_num[0]:])
            # corr_rna = pearson_correlation(reco_x, data[0])
            # corr_atac = pearson_correlation(reco_y, data[1])
            # print(f"Pearson 相关性 RNA: {corr_rna:.4f}")
            # print(f"Pearson 相关性 ATAC: {corr_atac:.4f}")

        self.log_dicts.append(metrics)
        print(f'最佳 ARI/foscttm: {best:.4f} 在第 {epoch_best} 轮')
        metric_names = ['foscttm', 'LTA', 'ARI', 'NMI', 'AMI', 'omics_mixing', 'bio_var_conser', 'accuracy']
        # for index, (name, value) in enumerate(zip(metric_names, metrics)):
        #     print(f"{name}= {value:.4f}, ")

        metrics_series = pd.Series(metrics, index=metric_names)
        print(f"最终模型结果：")
        for index, item in metrics_series.items():
            print(f"{index} = {item:.4f},")

    def test(self, data, graph_x, graph_y):
        self.model.eval()
        z_joint = self.model(data[0], graph_x[0],
                             data[1], graph_y[0],
                             self.epoch_best,
                             marginals_mode="uniform",
                             graph_mode="distance")
        return z_joint.cpu().detach().numpy()


def get_embedding(data, graph_x, graph_y, data_id, model_path, model, cell_num, device='cuda'):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"Loaded best model from {model_path}")

    model.eval()
    with torch.no_grad():
        total_loss, loss_dict, z_joint, reco_data = model(data[0], graph_x[0],
                                               data[1], graph_y[0], 0,
                                               marginals_mode="uniform",
                                               graph_mode="distance")
        inte = [z_joint[:cell_num[0]], z_joint[cell_num[0]:]]
        scGSi_inte = dict({"inte": inte})

        path = 'E:/experiment/scGSi_备份/results/' + data_id
        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, 'scGSi.npy'), scGSi_inte)
