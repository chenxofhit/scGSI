import torch
import random
import pandas as pd
import scanpy as sc
import numpy as np
from munkres import Munkres
from anndata import AnnData
import sklearn.neighbors
from scGSI.typehint import RandomState, get_rs
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics import davies_bouldin_score, accuracy_score, adjusted_rand_score, normalized_mutual_info_score, \
    adjusted_mutual_info_score, silhouette_score
from sklearn import metrics


def calc_frac_idx(z_rna, z_atac):
    """
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    n_sample = z_rna.shape[0]
    rank = 0
    for row_idx in range(n_sample):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(z_rna[row_idx, :], z_atac)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank = sort_euc_dist.index(true_nbr)
        frac = float(rank) / (n_sample - 1)

        fracs.append(frac)
        x.append(row_idx + 1)

    return fracs, x


def FOSCTTM(z_rna, z_atac):
    """
    来自 SCOT 的指标："FOSCTTM"
    输出平均 FOSCTTM 度量（在两个域上取平均）
    计算两个方向上所有数据点的匹配分数
    对每个数据点在两个方向上的分数取平均
    """
    z_rna = z_rna.detach().cpu().numpy() if isinstance(z_rna, torch.Tensor) else z_rna
    z_atac = z_atac.detach().cpu().numpy() if isinstance(z_atac, torch.Tensor) else z_atac
    fracs1, xs = calc_frac_idx(z_rna, z_atac)
    fracs2, xs = calc_frac_idx(z_atac, z_rna)

    avg_foscttm = (np.mean(fracs1) + np.mean(fracs2)) / 2

    return avg_foscttm


def LTA_acc(data1, data2, type1, type2, k_neighbors):
    """ clusters_label 衡量局部结构的保持程度。
    UnionCom的指标：“标签转移准确率”
    data1：第一个数据集嵌入矩阵，形状 (n_1, d)，n_1 为样本数，d 为嵌入维度。
    data2：第二个数据集嵌入矩阵，形状 (n_2, d)，n_2 为样本数，d 为嵌入维度。
    type1：第一个数据集的真实标签，形状 (n_1,)。
    type2：第二个数据集的真实标签，形状 (n_2,)。
    """
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(data2, type2)
    type1_predict = knn.predict(data1)
    knn.fit(data1, type1)
    type2_predict = knn.predict(data2)
    np.savetxt("E:/experiment/scGSi_备份/results/rna_Classifier_label_predict.txt", type1_predict)
    np.savetxt("E:/experiment/scGSi_备份/results/atac_Classifier_label_predict.txt", type2_predict)
    count = 0
    for label1, label2 in zip(type1_predict, type1):
        if label1 == label2:
            count += 1
    return count / len(type1)  # 单个数据集的LTA


def ari_nmi_ami_score1(clustering_rna_pred, true_labels):
    """
    计算 Adjusted Rand Index，衡量嵌入聚类与标签的一致性。
    参数:
        co_embedding (np.ndarray): 联合嵌入，形状 (n_cells, out_dim)。
        labels (np.ndarray): 数据集标签或细胞类型标签，形状 (n_cells,)。
        k (int): KMeans 聚类数（默认 20）。
    返回:
        score (float): ARI，范围 [-1, 1]，越高越好。
    """
    # 计算 ARI NMI
    ari = adjusted_rand_score(true_labels, clustering_rna_pred)
    nmi = normalized_mutual_info_score(true_labels, clustering_rna_pred)
    ami = adjusted_mutual_info_score(true_labels, clustering_rna_pred)
    return ari, nmi, ami


def ari_nmi_score(co_embedding, true_labels, n_rna, k_clusters):
    """
    计算 Adjusted Rand Index，衡量嵌入聚类与标签的一致性。
    参数:
        co_embedding (np.ndarray): 联合嵌入，形状 (n_cells, out_dim)。
        labels (np.ndarray): 数据集标签或细胞类型标签，形状 (n_cells,)。
        k (int): KMeans 聚类数（默认 20）。
    返回:
        score (float): ARI，范围 [-1, 1]，越高越好。
    """
    kmeans = KMeans(n_clusters=k_clusters, random_state=666)
    kmeans.fit_predict(co_embedding)
    clustering_pred_labels = kmeans.labels_
    # 分割 RNA 和 ATAC 的预测标签
    rna_pred = clustering_pred_labels[:n_rna]
    atac_pred = clustering_pred_labels[n_rna:]
    rna_labels = true_labels[:n_rna]
    atac_labels = true_labels[n_rna:]
    # 计算 ARI NMI
    ari_rna = adjusted_rand_score(rna_labels, rna_pred)
    ari_atac = adjusted_rand_score(atac_labels, atac_pred)
    nmi_rna = normalized_mutual_info_score(rna_labels, rna_pred)
    nmi_atac = normalized_mutual_info_score(atac_labels, atac_pred)
    np.savetxt("E:/experiment/scGSi_备份/result/clustering_label_predict.txt", clustering_pred_labels)

    # ari = adjusted_rand_score(true_labels, clustering_pred_labels)
    # nmi = normalized_mutual_info_score(true_labels, clustering_pred_labels)
    return ari_rna, ari_atac, nmi_rna, nmi_atac


# def nmi_score(co_embedding, labels, k=21):
#     """
#     计算 Normalized Mutual Information，衡量嵌入聚类与标签的信息共享。
#     参数:
#         co_embedding (np.ndarray): 联合嵌入，形状 (n_cells, out_dim)。
#         labels (np.ndarray): 数据集标签或细胞类型标签，形状 (n_cells,)。
#         k (int): KMeans 聚类数（默认 20）。
#     返回:
#         score (float): NMI，范围 [0, 1]，越高越好。
#     """
#     kmeans = KMeans(n_clusters=k, random_state=666)
#     pred_labels = kmeans.fit_predict(co_embedding)
#     score = normalized_mutual_info_score(labels, pred_labels)
#     return score


def get_k_neigh_ind(X, k_neighbors):
    neigh = NearestNeighbors(n_neighbors=k_neighbors)
    neigh.fit(X)
    neigh_dist, neigh_ind = neigh.kneighbors(X)
    return neigh_dist, neigh_ind


def neigh_overlap(z_rna, z_atac, k_neighbors):
    dsize = z_rna.shape[0]
    _, neigh_ind = get_k_neigh_ind(np.concatenate((z_rna, z_atac), axis=0), k_neighbors=k_neighbors)
    #     print(neigh_ind)
    z1_z2 = ((neigh_ind[:dsize, :] - dsize - np.arange(dsize)[:, None]) == 0)
    #     print(z1_z2)
    z2_z1 = (neigh_ind[dsize:, :] - np.arange(dsize)[:, None] == 0)
    #     print(z2_z1)
    return 0.5 * (np.sum(z1_z2) + np.sum(z2_z1)) / dsize


def graph_connectivity(x, y):
    r"""
    Graph connectivity

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    **kwargs
        Additional keyword arguments are passed to
        :func:`scanpy.pp.neighbors`

    Returns
    -------
    conn
        Graph connectivity
    """
    x = AnnData(X=x)
    sc.pp.neighbors(x, n_pcs=0, use_rep="X")
    conns = []
    for y_ in np.unique(y):
        x_ = x[y == y_]
        _, c = connected_components(
            x_.obsp['connectivities'],
            connection='strong'
        )
        counts = pd.Series(c).value_counts()
        conns.append(counts.max() / counts.sum())
    return np.mean(conns).item()


def seurat_alignment_score(x, y, neighbor_frac=0.01, n_repeats=4, random_state=None):
    r"""
    Seurat's alignment score

    Parameters
    ----------
    x
        Coordinates
    y
        Batch labels
    neighbor_frac
        Nearest neighbor fraction
    n_repeats
        Number of subsampling repeats
    random_state
        Random state
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    sas
        Seurat alignment score
    """
    rs = get_rs(random_state)
    idx_list = [np.where(y == u)[0] for u in np.unique(y)]
    min_size = min(idx.size for idx in idx_list)
    repeat_scores = []
    for _ in range(n_repeats):
        subsample_idx = np.concatenate([
            rs.choice(idx, min_size, replace=False)
            for idx in idx_list
        ])
        subsample_x = x[subsample_idx]
        subsample_y = y[subsample_idx]
        k = max(round(subsample_idx.size * neighbor_frac), 1)
        nn = NearestNeighbors(
            n_neighbors=k + 1).fit(subsample_x)
        nni = nn.kneighbors(subsample_x, return_distance=False)
        same_y_hits = (
                subsample_y[nni[:, 1:]] == np.expand_dims(subsample_y, axis=1)
        ).sum(axis=1).mean()
        repeat_score = (k - same_y_hits) * len(idx_list) / (k * (len(idx_list) - 1))
        repeat_scores.append(
            min(repeat_score, 1))  # score may exceed 1, if same_y_hits is lower than expected by chance
    return np.mean(repeat_scores).item()


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))

    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    clusterLabels = np.zeros(n_clusters)
    for i in range(n_clusters):
        clusterLabels[i] = indices[i][1]
    return clusterLabels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """Computes the predicted labels, where label assignments now
        correspond to the actual labels in y_true (as estimated by Munkres)

        Args:
            cluster_assignments: array of labels, outputted by kmeans
            y_true:              true labels
            n_clusters:          number of clusters in the dataset

        Returns:
            a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def accuracy_score1(co_embedding_np, cluster_labels_np, k_clusters):
    kmeans = KMeans(n_clusters=k_clusters, random_state=666).fit(
        co_embedding_np)
    kmeans_clustering_labels = kmeans.labels_
    y_pred_ajusted = get_y_preds(cluster_labels_np, kmeans_clustering_labels, k_clusters)
    acc = accuracy_score(cluster_labels_np, y_pred_ajusted)

    return acc


def ari_nmi_ami_score(co_embedding, true_labels, k):
    """
    计算 Adjusted Rand Index，衡量嵌入聚类与标签的一致性。
    参数:
        co_embedding (np.ndarray): 联合嵌入，形状 (n_cells, out_dim)。
        labels (np.ndarray): 数据集标签或细胞类型标签，形状 (n_cells,)。
        k (int): KMeans 聚类数（默认 20）。
    返回:
        score (float): ARI，范围 [-1, 1]，越高越好。
    """
    kmeans = KMeans(n_clusters=k, random_state=666)
    kmeans.fit_predict(co_embedding)
    clustering_pred_labels = kmeans.labels_
    # 分割 RNA 和 ATAC 的预测标签
    # rna_pred = clustering_pred_labels[:n_rna]
    # atac_pred = clustering_pred_labels[n_rna:]
    # rna_labels = true_labels[:n_rna]
    # atac_labels = true_labels[n_rna:]
    # 计算 ARI NMI
    ari_rna = adjusted_rand_score(true_labels, clustering_pred_labels)
    # ari_atac = adjusted_rand_score(atac_labels, atac_pred)
    nmi_rna = normalized_mutual_info_score(true_labels, clustering_pred_labels)
    # nmi_atac = normalized_mutual_info_score(atac_labels, atac_pred)
    ami_rna = adjusted_mutual_info_score(true_labels, clustering_pred_labels)
    # ami_atac = adjusted_mutual_info_score(atac_labels, atac_pred)
    return ari_rna, nmi_rna, ami_rna


def mean_average_precision(x, y, neighbor_frac=0.01, **kwargs):
    r"""
    Mean average precision

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels
    neighbor_frac
        Nearest neighbor fraction
    **kwargs
        Additional keyword arguments are passed to
        :class:`sklearn.neighbors.NearestNeighbors`

    Returns
    -------
    map
        Mean average precision
    """
    k = max(round(y.shape[0] * neighbor_frac), 1)
    nn = NearestNeighbors(
        n_neighbors=min(y.shape[0], k + 1), **kwargs
    ).fit(x)
    nni = nn.kneighbors(x, return_distance=False)
    match = np.equal(y[nni[:, 1:]], np.expand_dims(y, 1))
    return np.apply_along_axis(_average_precision, 1, match).mean().item()


def _average_precision(match: np.ndarray) -> float:
    if np.any(match):
        cummean = np.cumsum(match) / (np.arange(match.size) + 1)
        return cummean[match].mean().item()
    return 0.0


def knn_purity_score(embedding, labels, k_neighbors):
    """Compute the kNN purity score, averaged over all observations.
    For one observation, the purity score is the percentage of
    nearest neighbors that share its label.
    from Mowgli
    Args:
        embedding:

        knn (np.ndarray):
            The knn, shaped (n_obs, k). The i-th row should contain integers
            representing the indices of the k nearest neighbors.
        labels (np.ndarray):
            The labels, shaped (n_obs)
        k_neighbors:

    Returns:
        float: The purity score.
        :param k_neighbors:
        :param labels:
        :param embedding:
    """
    knn = embedding_to_knn(embedding, k=k_neighbors, metric="euclidean")
    assert knn.shape[0] == labels.shape[0]

    # Initialize a list of purity scores.
    score = 0

    # Iterate over the observations.
    for i, neighbors in enumerate(knn):
        # Do the neighbors have the same label as the observation?
        matches = labels[neighbors] == labels[i]

        # Add the purity rate to the scores.
        score += np.mean(matches) / knn.shape[0]

    return score


def embedding_to_knn(embedding, k, metric="euclidean"):
    """Convert embedding to knn

    Args:
        embedding (np.ndarray): The embedding (n_obs, n_latent)
        k (int, optional): The number of nearest neighbors. Defaults to 21.
        metric (str, optional): The metric to compute neighbors with. Defaults to "euclidean".

    Returns:
        np.ndarray: The knn (n_obs, k)
    """
    # Initialize the knn graph.
    knn = np.zeros((embedding.shape[0], k), dtype=int)

    # Compute pariwise distances between observations.
    distances = cdist(embedding, embedding, metric=metric)

    # Iterate over observations.
    for i in range(distances.shape[0]):
        # Get the `max_neighbors` nearest neighbors.
        knn[i] = distances[i].argsort()[1: k + 1]

    # Return the knn graph.
    return knn


def avg_silhouette_width(x, y):
    r"""
    Cell type average silhouette width

    Parameters
    ----------
    x
        Coordinates
    y
        Cell type labels

    Returns
    -------
    asw
        Cell type average silhouette width

    Note
    ----
    Follows the definition in `OpenProblems NeurIPS 2021 competition
    <https://openproblems.bio/neurips_docs/about_tasks/task3_joint_embedding/>`__
    """
    return (silhouette_score(x, y) + 1) / 2


def evaluate_alignment(z_rna, z_atac, labels, k_clusters, k_neighbors):
    """
    评估对齐性能，计算多种指标。
    参数:
        z_rna (torch.Tensor or np.ndarray): RNA嵌入，形状 (n_rna, out_dim)。
        z_atac (torch.Tensor or np.ndarray): ATAC嵌入，形状 (n_atac, out_dim)。
        labels (list): [RNA标签, ATAC标签]，数据集标签或细胞类型标签。
        data (list): [RNA数据, ATAC数据]，输入的原始数据。
        reco_rna (torch.Tensor): 重构的RNA数据，形状 (n_cells_rna, gene_dim)。
        reco_atac (torch.Tensor): 重构的ATAC数据，形状 (n_cells_atac, peak_dim)。
        k_clusters (int): KMeans聚类数（默认 20）。
        k_neighbors (int): kNN邻居数（默认 5）。
    返回:
        metrics (dict): 包含 FOSCTTM, Label Accuracy, Silhouette Score, ARI, NMI, kNN Alignment Score, FID-like Distance, MSE RNA, MSE ATAC。
    """
    # ari = adjusted_rand_score(labels, pred)
    # nmi = normalized_mutual_info_score(labels, pred)

    # 统一数据类型转换
    z_rna_np = z_rna.detach().cpu().numpy() if isinstance(z_rna, torch.Tensor) else z_rna
    z_atac_np = z_atac.detach().cpu().numpy() if isinstance(z_atac, torch.Tensor) else z_atac
    rna_clusters_label_np = labels[0].cpu().numpy() if isinstance(labels[0], torch.Tensor) else labels[0]
    atac_clusters_label_np = labels[1].cpu().numpy() if isinstance(labels[1], torch.Tensor) else labels[1]

    # 合并嵌入和标签
    n_rna, n_atac = z_rna_np.shape[0], z_atac_np.shape[0]
    co_embedding_np = np.concatenate([z_rna_np, z_atac_np], axis=0)  # 形状 (n_rna + n_atac, out_dim)
    cluster_labels_np = np.concatenate([rna_clusters_label_np, atac_clusters_label_np], axis=0)  # 合并聚类标签
    data_source_labels = torch.zeros(n_rna + n_atac, dtype=torch.long,
                                     device=z_rna.device if isinstance(z_rna, torch.Tensor) else 'cpu')
    data_source_labels[n_rna:] = 1  # RNA: 0, ATAC: 1
    data_source_labels_np = data_source_labels.cpu().numpy()
    # co_embedding_torch = torch.from_numpy(co_embedding_np).to(data_source_labels.device)

    # 计算指标
    foscttm = FOSCTTM(z_rna_np, z_atac_np)
    label_acc = LTA_acc(z_rna_np, z_atac_np, rna_clusters_label_np, atac_clusters_label_np, k_neighbors)
    rna_pred_labels = np.loadtxt("E:/experiment/scGSi_备份/results/rna_Classifier_label_predict.txt")
    atac_pred_labels = np.loadtxt("E:/experiment/scGSi_备份/results/atac_Classifier_label_predict.txt")
    cluster_pred_labels = np.hstack((rna_pred_labels, atac_pred_labels))
    # ari, nmi, ami = ari_nmi_ami_score(co_embedding_np, cluster_labels_np, k=k_clusters)

    # ari_rna, ari_atac, nmi_rna, nmi_atac = ari_nmi_score(co_embedding_np, cluster_labels_np, n_rna, k=k_clusters)
    ari, nmi, ami = ari_nmi_ami_score1(rna_pred_labels, rna_clusters_label_np)
    # fid_like = fid_like_distance(co_embedding_np, data_source_labels.cpu().numpy())
    neighborhood_overlap_score = neigh_overlap(z_rna_np, z_atac_np, k_neighbors)
    graph_conn = graph_connectivity(co_embedding_np, cluster_labels_np)
    seurat_align_score = seurat_alignment_score(co_embedding_np, data_source_labels_np, random_state=0)

    omics_mixing = (neighborhood_overlap_score + graph_conn + seurat_align_score) / 3

    asw = avg_silhouette_width(co_embedding_np, cluster_labels_np)
    purity = knn_purity_score(co_embedding_np, cluster_labels_np, k_neighbors)
    mapre = mean_average_precision(co_embedding_np, cluster_labels_np)

    # 生物变异保留评估整合后数据集的保存留情况
    bio_var_conser = (asw + purity + mapre) / 3

    accuracy = accuracy_score1(co_embedding_np, cluster_labels_np, k_clusters)
    # accuracy = accuracy_score(cluster_labels_np, cluster_pred_labels)

    metrics = [foscttm, label_acc, ari, nmi, ami, omics_mixing, bio_var_conser,accuracy]

    return metrics


def test_alignment_score(data1_shared, data2_shared, data1_specific=None, data2_specific=None):
    N = 2

    if len(data1_shared) < len(data2_shared):
        data1 = data1_shared
        data2 = data2_shared
    else:
        data2 = data1_shared
        data1 = data2_shared
    data2 = data2[random.sample(range(len(data2)), len(data1))]
    k = np.maximum(10, (len(data1) + len(data2)) * 0.01)
    k = k.astype(np.int)

    data = np.vstack((data1, data2))

    bar_x1 = 0
    for i in range(len(data1)):
        diffMat = data1[i] - data
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k + 1]
        for j in NearestN:
            if j < len(data1):
                bar_x1 += 1
    bar_x1 = bar_x1 / len(data1)

    bar_x2 = 0
    for i in range(len(data2)):
        diffMat = data2[i] - data
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k + 1]
        for j in NearestN:
            if j >= len(data1):
                bar_x2 += 1
    bar_x2 = bar_x2 / len(data2)

    bar_x = (bar_x1 + bar_x2) / 2

    score = 0
    score += 1 - (bar_x - k / N) / (k - k / N)

    data_specific = None
    flag = 0
    if data1_specific is not None:
        data_specific = data1_specific
        if data2_specific is not None:
            data_specific = np.vstack((data_specific, data2_specific))
            flag = 1
    else:
        if data2_specific is not None:
            data_specific = data2_specific

    if data_specific is None:
        return score
    else:
        bar_specific1 = 0
        bar_specific2 = 0
        data = np.vstack((data, data_specific))
        if flag == 0:  # only one of data1_specific and data2_specific is not None
            for i in range(len(data_specific)):
                diffMat = data_specific[i] - data
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k + 1]
                for j in NearestN:
                    if j > (len(data1) + len(data2)):
                        bar_specific1 += 1
            bar_specific = bar_specific1

        else:  # both data1_specific and data2_specific are not None
            for i in range(len(data1_specific)):
                diffMat = data1_specific[i] - data
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k + 1]
                for j in NearestN:
                    if (len(data1) + len(data2)) < j < (len(data1) + len(data2) + len(data1_specific)):
                        bar_specific1 += 1

            for i in range(len(data2_specific)):
                diffMat = data2_specific[i] - data
                sqDiffMat = diffMat ** 2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k + 1]
                for j in NearestN:
                    if j > (len(data1) + len(data2) + len(data1_specific)):
                        bar_specific2 += 1

            bar_specific = bar_specific1 + bar_specific2

        bar_specific = bar_specific / len(data_specific)

        score += (bar_specific - k / N) / (k - k / N)

        return score / 2
