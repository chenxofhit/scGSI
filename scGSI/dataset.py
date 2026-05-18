import os
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scGSi.preprocess import preprocess_rna, preprocess_atac


def load_data(rna_path, atac_path, batch_size=256, device='cuda', n_pca=50, n_lsa=50):
    if not (os.path.exists(rna_path) and os.path.exists(atac_path)):
        raise FileNotFoundError(f"RNA path {rna_path} or ATAC path {atac_path} does not exist.")

    rna_pca, rna_raw, adata_rna = preprocess_rna(sc.read_h5ad(rna_path), n_components=n_pca, device=device)
    atac_lsa, atac_raw, adata_atac = preprocess_atac(sc.read_h5ad(atac_path), n_components=n_lsa, device=device)

    if rna_pca.shape[0] == 0 or atac_lsa.shape[0] == 0:
        raise ValueError("Preprocessed data is empty.")
    # 检查 labels 列是否存在
    if 'labels' not in adata_rna.obs.columns:
        raise ValueError("There is no 'labels' column in adata_rna.obs")
    if 'labels' not in adata_atac.obs.columns:
        raise ValueError("There is no 'labels' column in adata_atac.obs")
    rna_dataset = rnaDataset(rna_pca, rna_raw,
                             adata_rna.obs.get('labels', device=device))
    atac_dataset = atacDataset(atac_lsa, atac_raw,
                               adata_atac.obs.get('labels', device=device))

    rna_dataloader = DataLoader(rna_dataset, batch_size=batch_size, shuffle=True,
                                collate_fn=rnaCollate(device=device))
    atac_dataloader = DataLoader(atac_dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=atacCollate(device=device))

    return rna_dataloader, atac_dataloader, rna_pca.shape[1], atac_lsa.shape[1], rna_raw, atac_raw


class rnaDataset(Dataset):

    def __init__(self, data_train, data_test, label):

        if not isinstance(data_train, torch.Tensor) or not isinstance(data_test, torch.Tensor):
            raise TypeError("data_fit and data_supervision must be torch.Tensor.")
        if data_train.shape[0] != data_test.shape[0]:
            raise ValueError(
                f"data_fit ({data_train.shape[0]}) and data_supervision ({data_test.shape[0]}) must have same number of cells.")
        if label.shape[0] != data_train.shape[0]:
            raise ValueError(
                f"label ({label.shape[0]}) must match data_fit ({data_train.shape[0]}) in number of cells.")

        self.data_train = data_train
        self.data_test = data_test
        self.label = label

    def __len__(self):
        return self.data_train.shape[0]

    def __getitem__(self, index):
        return self.data_train[index], self.data_test[index], self.label[index]


class atacDataset(Dataset):

    def __init__(self, data_train, data_test, label):
        assert (data_test is None or len(data_train) == len(data_test)) and len(data_train) == len(label)
        self.data_train = data_train
        self.data_test = data_test
        self.label = label

    def __len__(self):
        return self.data_train.shape[0]

    def __getitem__(self, index):
        if self.data_test is None:
            return self.data_train[index], None, self.label[index]
        else:
            return self.data_train[index], self.data_test[index], self.label[index]


class rnaCollate:
    # 统一批处理 为fit和supervision数据 并将数据转换为 torch.Tensor
    # 命名为Collate还是分别单独命名，考虑合并还是分开

    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, batch):
        data_train, data_test, batch_label = zip(*batch)
        data_train = torch.stack([torch.as_tensor(x, dtype=torch.float,
                                                  device=self.device) for x in data_train])
        data_test = torch.stack([torch.as_tensor(x, dtype=torch.float,
                                                 device=self.device) if x is not None else torch.zeros_like(
            data_train[0]) for x in data_test])
        batch_label = torch.stack(
            [torch.as_tensor(x, dtype=torch.long, device=self.device) for x in batch_label]).squeeze(-1)
        return data_train, data_test, batch_label

    # 考虑是否增设校准数据集，然后增加使用分类器来从共同嵌入中来进行预测分类标签


class atacCollate:

    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, batch):
        data_train, data_test, batch_label = zip(*batch)
        data_train = torch.from_numpy(np.vstack(data_train)).to(self.device)
        if data_test[0] is not None:
            data_test = torch.from_numpy(np.vstack(data_test)).to(self.device)

        return data_train, data_test
