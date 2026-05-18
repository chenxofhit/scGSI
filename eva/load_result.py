import torch
import numpy as np
import anndata
from sklearn.preprocessing import LabelEncoder
import os


def load_clusters(anno_atac, anno_rna):

    all_cell_types = list(set(anno_atac.tolist() + anno_rna.tolist()))
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(all_cell_types)
    clu_atac = label_encoder.transform(anno_atac)
    clu_rna = label_encoder.transform(anno_rna)

    return clu_atac, clu_rna


def convert_to_numpy(tensor_or_array):
    """
    Convert input to numpy array.
    If it's a torch.Tensor, move to CPU and convert to numpy.
    If it's already a numpy.ndarray, return as is.
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    else:
        raise ValueError(f"Unsupported type: {type(tensor_or_array)}. Expected torch.Tensor or np.ndarray.")


def load_result(dataset_name, dataset_dir, result_dir, methods):

    RNA_data = anndata.read_h5ad(os.path.join(dataset_dir, dataset_name + '/raw_data_rna.h5ad'))
    ATAC_data = anndata.read_h5ad(os.path.join(dataset_dir, dataset_name + '/raw_data_atac.h5ad'))
    RNA_data.obs['omic_id'] = 'RNA'
    ATAC_data.obs['omic_id'] = 'ATAC'

    anno_rna, anno_atac = RNA_data.obs['cell_type'], ATAC_data.obs['cell_type']
    clu_atac, clu_rna = load_clusters(anno_atac, anno_rna)
    num_atac, num_rna = len(clu_atac), len(clu_rna)
    adata = anndata.concat([RNA_data, ATAC_data], join='outer')
    adata.obs['cell_type'] = np.concatenate((anno_rna, anno_atac), axis=0)
    adata.obs['cluster'] = np.concatenate((clu_rna, clu_atac), axis=0)
    adata.obs['omic_id'] = np.concatenate((RNA_data.obs['omic_id'], ATAC_data.obs['omic_id']), axis=0)

    for i in range(len(methods)):
        method = methods[i]

        if method == 'raw_data':
            adata.obsm[method] = np.vstack((RNA_data.X, ATAC_data.X))

        # elif method == 'scMGPF' or method == 'MMDMA' or method == 'JointMDS':
        #     result = np.load(os.path.join(result_dir, method + ".npy"), allow_pickle=True).item()
        #     adata.obsm[method] = np.vstack(
        #         (result['inte'][0].cpu().numpy(), result['inte'][1].cpu().numpy()))  # rna, atac
        # else:
        #     result = np.load(os.path.join(result_dir, method + ".npy"), allow_pickle=True).item()
        #     adata.obsm[method] = np.vstack((result['inte'][0], result['inte'][1]))
        else:
            result = np.load(os.path.join(result_dir, method + ".npy"), allow_pickle=True).item()
            rna_inte = convert_to_numpy(result['inte'][0])
            atac_inte = convert_to_numpy(result['inte'][1])
            adata.obsm[method] = np.vstack((rna_inte, atac_inte))
    return adata, anno_rna, anno_atac, clu_rna, clu_atac

 