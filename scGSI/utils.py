import random
import time
import yaml
import json
import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from torch import block_diag
from scipy.sparse import diags, coo_matrix, block_diag
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_undirected, degree

# For computing graph distances:
from sklearn.neighbors import kneighbors_graph


def init_random_seed(manual_seed):
    if manual_seed is None:
        seed = int(time.time() * 1000) % (2 ** 32)  # 基于时间生成种子
    else:
        if not isinstance(manual_seed, int) or manual_seed < 0:
            raise ValueError("manual_seed must be a non-negative integer.")
        seed = manual_seed

    print(f"Using random seed: {seed}")
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def load_config(config_name, verbose=True):
    if '.yaml' not in config_name:
        config_name += '.yaml'
    with open(config_name, 'r') as f:
        f_str = f.read()
        dic = yaml.safe_load(f_str)
        if verbose:
            js = json.dumps(dic, sort_keys=True, indent=4, separators=(',', ':'))
            print(js)
        return dic


def check_balance(data, cell_ratio_thresholds=1.1, feature_ratio_thresholds=2.0):
    """
    评估 RNA 和 ATAC 数据的平衡性，包括细胞和特征数量。
        参数:
            cell_balance_status : 细胞平衡性状态 ， 默认true
            feature_balance_status : 特征平衡性状态 ， 默认true
            cell_ratio_thresholds : 细胞数量比率阈值 ，默认 1.1。
            feature_ratio_thresholds : 特征数量比率阈值 ，默认 2.0。
        返回:
            dict: 包含不平衡指标及其他指标
    """
    if len(data) < 2:
        raise ValueError("At least two datasets  are required.")

    result = {
        'cell_details': {},
        'feature_details': {},
        'cell_balance_status': True,
        'feature_balance_status': True,
    }

    # 细胞数量分析
    cell_counts = [data.shape[0] for data in data]
    cell_min = min(cell_counts)
    cell_ratio = max(cell_counts) / cell_min if cell_min > 0 else float('inf')

    result['cell_details'] = {
        'rna_cells': cell_counts[0],
        'atac_cells': cell_counts[1],
        'ratio': cell_ratio
    }

    result['cell_balance_status'] = cell_ratio <= cell_ratio_thresholds

    # 特征数量分析
    feature_counts = [data.shape[1] for data in data]
    feature_min = min(feature_counts)
    feature_ratio = max(feature_counts) / feature_min if feature_min > 0 else float('inf')

    result['feature_details'] = {
        'rna_features': feature_counts[0],
        'atac_features': feature_counts[1],
        'ratio': feature_ratio
    }

    result['feature_balance_status'] = feature_ratio <= feature_ratio_thresholds

    return result


class Construct_Graph_and_intra_distances:

    def __init__(self, data, cell_num, device="cuda"):
        self.data = data  # 数据矩阵：RNA (N × G, 连续值) 或 ATAC (N × P, 二值)
        self.device = device
        self.graphs = []  # 每个数据集的图
        self.disgraph_x = None  # intra-datasets graph distances for datasets 1 (X)
        self.disgraph_y = None  # intra-datasets graph distances for datasets 2 (y)
        self.intra_graphDists = []  # Holds intra-domain graph distances for each input dataset
        self.cell_num = cell_num

    def construct_knn_graph(self, graph_mode, k, metric="minkowski", p=2, return_edge_index=True):
        """有向图
        参数:
            k (int): 邻居数量。
            mode (str): "connectivity"（二值邻接矩阵）或 "distance"（加权邻接矩阵）。
            metric (str): 距离度量，可选 "minkowski"（默认 p=2 为欧几里得距离）、"jaccard"、"correlation"。
            p (float): 明可夫斯基距离的指数，仅当 metric="minkowski" 时有效。
            return_edge_index (bool): 是否返回 PyTorch Geometric 的 edge_index 格式。
        返回:
             list: 每个数据集的 KNN 图   其中graphs[0]为元组（边索引或 稀疏矩阵）。
        """
        assert graph_mode in ["connectivity", "distance"], "Mode argument must be 'connectivity' or 'distance'."
        assert metric in ["minkowski", "jaccard",
                          "correlation"], "Metric must be 'minkowski', 'jaccard', or 'correlation'."
        idx = 0 if metric == "minkowski" else 1
        if len(self.data[idx].shape) != 2:
            raise ValueError(f"Data is not a 2D matrix.")

        for i, data in enumerate(self.data):
            if isinstance(data, torch.Tensor):
                X = data.cpu().numpy()  # 从 GPU 移动到 CPU 并转换为 NumPy
            else:
                X = np.asarray(data)  # 确保是 NumPy 数组
                # 为 Jaccard 距离转换数据为布尔类型
            if metric == "jaccard" and X.dtype != np.bool_:
                X = (X != 0).astype(np.bool_)  # 转换为布尔类型以适配 Jaccard 度量

            # 初始化 NearestNeighbors
            if metric == "minkowski":

                nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, p=p)  # 不包括自身数据点，n_neighbors=k + 1为包括自身数据点

            else:
                nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
            # 计算 KNN图
            nbrs.fit(X)
            if graph_mode == "distance":
                distances, indices = nbrs.kneighbors(X)

                if indices.shape[1] != distances.shape[1]:
                    raise ValueError(f"Edge index and weight mismatch for data")
                # 构建加权邻接矩阵
                rows = np.repeat(np.arange(X.shape[0]), k)  # 表示边起点
                indices = indices[:, 1:]
                distances = distances[:, 1:]  # 跳过自身数据点（第一个邻居）
                cols = indices.flatten()  # 表示边终点
                distances = distances.flatten()
                if return_edge_index:
                    # edge_index 是一个形状为 (2, num_edges) 的张量
                    # edge_weight 是一个形状为 (num_edges,) 的张量
                    edge_index_np = np.array([rows, cols], dtype=np.int64)
                    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=self.device)
                    # edge_index = to_undirected(edge_index, num_nodes=X.shape[0])
                    edge_weight = torch.tensor(distances, dtype=torch.float, device=self.device)
                    self.graphs.append((edge_index, edge_weight))
                else:
                    graph = sp.csr_matrix((distances, (rows, cols)), shape=(X.shape[0], X.shape[0]))  # 稀疏的加权邻接矩阵
                    self.graphs.append(graph)

            else:  # mode == "connectivity"
                graph = nbrs.kneighbors_graph(X, mode="connectivity")  # 稀疏的连通矩阵
                if return_edge_index:
                    rows, cols = graph.nonzero()
                    edge_index = torch.tensor([rows, cols], dtype=torch.long, device=self.device)
                    edge_index = to_undirected(edge_index)
                    self.graphs.append((edge_index, None))  # 无权重
                else:
                    self.graphs.append(graph)
                # # 构建边索引（排除自身）
                #  rows = np.repeat(np.arange(X.shape[0]), k)
                #  cols = indices[:, 1:].flatten()  # 跳过自身数据点（第一个邻居）
                #  edge_index = torch.tensor([rows, cols], dtype=torch.long)     #  edge_index形状为(2,num_edges)，其中num_edges边数 =数据点数*k。
                #
                #  edge_index = to_undirected(edge_index)   # 转换为无向图
                # # KNN图列表，格式为(edge_index, edge_weight)

        return self.graphs

    def graph_distances_matrix_optimized(self):
        for i, graph_tuple in enumerate(self.graphs):
            # 假设图现在是SciPy稀疏矩阵
            if isinstance(graph_tuple, tuple):  # 如果仍使用PyG格式
                rows, cols = graph_tuple[0].cpu().numpy()
                weights = graph_tuple[1].cpu().numpy()
                num_nodes = self.cell_num[i]
                graph = csr_matrix((weights, (rows, cols)), shape=(num_nodes, num_nodes))
            else:
                graph = graph_tuple

            # 高效计算所有节点对之间的最短路径
            dist_matrix = dijkstra(csgraph=graph, directed=False)

            # 处理不连通组件产生的无穷大值
            inf_val = np.max(dist_matrix[np.isfinite(dist_matrix)])
            dist_matrix[np.isinf(dist_matrix)] = inf_val

            # 归一化并转换为Tensor
            dist_matrix /= dist_matrix.max()
            np.fill_diagonal(dist_matrix, 0)

            self.intra_graphDists.append(torch.tensor(dist_matrix, dtype=torch.float, device=self.device))

        return self.intra_graphDists

    def graph_distances_matrix(self):

        for i, graph in enumerate(self.graphs[0:1]):
            if not isinstance(graph, tuple) or len(graph) != 2:
                raise ValueError(f"Graph[{i}] must be a tuple of (edge_index, edge_weight).")
            edge_index, edge_weight = graph
            if not isinstance(edge_index, torch.Tensor) or not isinstance(edge_weight, torch.Tensor):
                raise ValueError(f"edge_index and edge_weight for Graph[{i}] must be torch.Tensor.")

            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)

            if edge_index.shape[1] == 0 or self.cell_num == 0:
                raise ValueError(f"Graph[{i}] is empty or has no nodes.")

            # 初始化距离矩阵（无穷大表示未连接）
            inf = float('inf')
            dist_matrix = torch.full((self.cell_num[i], self.cell_num[i]), inf, device=self.device)
            dist_matrix[torch.arange(self.cell_num[i]), torch.arange(self.cell_num[i])] = 0

            # 填充边权重
            row, col = edge_index
            dist_matrix[row, col] = edge_weight
            dist_matrix[col, row] = edge_weight  # 确保无向图

            # Floyd-Warshall
            for k in range(self.cell_num[i]):
                dist_matrix = torch.min(
                    dist_matrix,
                    dist_matrix[:, k].unsqueeze(1) + dist_matrix[k, :].unsqueeze(0)
                )

            max_dist = torch.max(dist_matrix[dist_matrix != inf])
            if torch.isnan(max_dist) or max_dist == 0:
                max_dist = torch.tensor(1.0, device=self.device)  # 避免除零
            dist_matrix[dist_matrix == inf] = max_dist

            dist_matrix = dist_matrix / (dist_matrix.max() + 1e-10)
            dist_matrix.fill_diagonal_(0)  # 确保对角线为 0

            self.intra_graphDists.append(dist_matrix)

        return self.intra_graphDists


def get_spatial_distance_matrix(data, metric="euclidean"):
    Cdata = sp.spatial.distance.cdist(data, data, metric=metric)
    return Cdata / Cdata.max()


def get_marginals(data, marginals_mode="uniform", knn_edge_indices=None, metadata=None, normalization=True,
                  device="cuda"):
    """
    参数:
        mode (str): 分布类型，选项：
            - "uniform": 均匀分布 (1/num_cells)。
            - "degree": 基于 KNN 图度数，权重正比于邻居数。
            - "expression": 基于总表达量（RNA）或峰值计数（ATAC）。
            - "metadata": 基于元数据（如细胞类型权重）。
        knn_edge_indices (list, optional): KNN 图边索引列表，用于 degree 模式。
        normalization (bool): 是否归一化分布（总和为 1）。

    返回:
        list: 边缘分布列表，每个元素为形状 (num_cells,) 的张量。(n,0)和（m,0)
    """
    if data is None:
        raise ValueError("self.data is empty.")

    marginals = []

    for i, data in enumerate(data):
        if len(data.shape) != 2:
            raise ValueError(f"Data[{i}] is not a 2D matrix.")
        num_cells = data.shape[0]
        if num_cells <= 0:
            raise ValueError(f"Data[{i}] has invalid number of cells.")

        if marginals_mode == "uniform":
            # 均匀分布 无先验
            marginal_dist = torch.ones(num_cells, device=device) / num_cells

        elif marginals_mode == "degree":
            # 基于 KNN 图度数 度数高的细胞（邻居多）分配更高权重，适合捕获拓扑结构。
            if knn_edge_indices is None or i >= len(knn_edge_indices):
                raise ValueError(f"KNN edge index for data[{i}] is required for degree mode.")
            edge_index = knn_edge_indices[i].to(device)
            deg = degree(edge_index[i][0], num_nodes=num_cells, dtype=torch.float)
            marginal_dist = deg / (deg.sum() + 1e-8)  # 归一化度数

        elif marginals_mode == "expression":
            # 基于总表达量（RNA）或峰值计数（ATAC）  活性差异
            expr_sum = data.sum(dim=1)  # 每细胞的总特征值
            marginal_dist = expr_sum / (expr_sum.sum() + 1e-8)  # 归一化

        elif marginals_mode == "metadata":
            # 基于元数据（如细胞类型权重）
            if metadata is None or i >= len(metadata):
                raise ValueError(f"Metadata for data[{i}] is required for metadata mode.")
            # 假设 metadata[i] 是细胞类型权重（例如，稀有细胞类型更高权重）
            weights = torch.tensor(metadata[i], dtype=torch.float, device=device)
            marginal_dist = weights / (weights.sum() + 1e-8)

        else:
            raise ValueError(
                f"Unsupported mode: {marginals_mode}. Choose from 'uniform', 'degree', 'expression', 'metadata'.")

        # 确保非负且归一化
        if normalization:
            marginal_dist = torch.clamp(marginal_dist, min=0)
            marginal_dist = marginal_dist / (marginal_dist.sum() + 1e-8)

        marginals.append(marginal_dist)

    return marginals

