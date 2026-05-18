import os
import torch
import numpy as np
import scanpy as sc
import scipy.sparse
import sklearn
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD

np.random.seed(666)


def preprocess_rna(adata_rna, target_sum=1e4, n_top_genes=2000, n_components=50, device='cuda'):
    if adata_rna is None or adata_rna.n_obs == 0 or adata_rna.n_vars == 0:
        raise ValueError("RNA-seq数据为空，无法处理")

    print(f"原始RNA-seq数据维度：{adata_rna.shape}")

    # 创建副本以避免修改原始数据
    adata = adata_rna.copy()
    adata.var_names_make_unique()

    if isinstance(adata.X, scipy.sparse.spmatrix):
        x_array = adata.X.toarray()
    else:
        x_array = adata.X.copy()
    if not np.all(np.modf(x_array)[0] == 0):
        print("检测到非整数值，数据可能已标准化")

    # 特征选择
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor='seurat_v3',
        subset=True
    )

    # 标准化
    sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=True)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=True, max_value=10)

    # 降维
    sc.pp.pca(adata, n_comps=n_components, zero_center=True, svd_solver='arpack')

    print(f"RNA-seq预处理完成，PCA结果：{adata.obsm['X_pca'].shape}")
    pca_rna_data = torch.tensor(adata.obsm['X_pca'], dtype=torch.float, device=device)
    # raw_data = torch.tensor(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X, dtype=torch.float,
    #                         device=device)

    return pca_rna_data, adata


def preprocess_atac(adata_atac, n_components=50, use_highly_variable=None, device='cuda'):
    if adata_atac is None or adata_atac.n_obs == 0 or adata_atac.n_vars == 0:
        raise ValueError("ATAC-seq数据为空，无法处理")

    print(f"原始ATAC-seq数据维度：{adata_atac.shape}")

    # 创建副本
    adata = adata_atac.copy()
    adata.var_names_make_unique()

    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    # LSI 降维：TF-IDF + SVD
    X = scipy.sparse.csr_matrix(adata_use.X)  # 确保稀疏格式

    # Term Frequency (TF): log(1 + counts)
    tf = X.multiply(1 / X.sum(axis=1))
    # Inverse Document Frequency (IDF): log(1 + n_cells / n_peaks)
    n_peaks_per_cell = X.getnnz(axis=0)
    idf = X.shape[0] / X.sum(axis=0)
    # TF-IDF 矩阵
    tfidf = tf.multiply(idf)
    tfidf = normalize(tfidf, norm="l1")  # 可尝试l2或者使用preprocess.py里的normalize_data
    tfidf = np.log1p(tfidf * 1e4)
    # SVD 降维
    lsi = sklearn.utils.extmath.randomized_svd(tfidf, n_components)[0]
    lsi -= lsi.mean(axis=1, keepdims=True)
    lsi /= lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm['X_lsi'] = lsi

    print(f"ATAC-seq预处理完成，LSA结果：{adata.obsm['X_lsi'].shape}")
    lsa_atac_data = torch.tensor(adata.obsm['X_lsi'], dtype=torch.float, device=device)
    # raw_data = torch.tensor(adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X, dtype=torch.float,
    #                         device=device)
    return lsa_atac_data, adata


def normalize_data(data, norm="l2", bySample=True):  # l2（欧几里得范数）归一化
    assert (norm in ["l1", "l2", "max",
                     "zscore"]), ("Norm argument has to be either one of 'max', 'l1', 'l2' or 'zscore'. If you would "
                                  "like to perform another type of normalization, please give SCOT the normalize data "
                                  "and set the argument normalize=False when running the algorithm.")

    for i in range(len(data)):
        if norm == "zscore":
            scaler = StandardScaler()
            data[i] = scaler.fit_transform(
                data[i])  # self.X, self.y = scaler.fit_transform(self.X), scaler.fit_transform(self.y)
        else:
            if (bySample == True or bySample is None):
                axis = 1
            else:
                axis = 0
            data[i] = normalize(data[i], norm=norm,
                                axis=axis)  # self.X,self.y=normalize(self.X,norm=norm,axis=axis),normalize(self,y,norm=norm,axis=axis)
    return data  # Normalized data


