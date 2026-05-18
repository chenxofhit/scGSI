from scGSI.preprocess import *
import pandas as pd
import anndata
from sklearn.decomposition import TruncatedSVD
import os
import sys
import scanpy as sc

from data.data_preprocess import rna_pca, atac_lsi

current_directory = os.path.dirname(os.path.abspath(__file__))

data_id = "Tea_PBMC"


dataset_dir = "E:/data/raw_data/Tea_PBMC"
RNA_data = anndata.read_h5ad(os.path.join(dataset_dir, 'RNA/test_RNA.h5ad'))
ATAC_data = anndata.read_h5ad(os.path.join(dataset_dir, 'ATAC/test_ATAC.h5ad'))


RNA_data.obs['batch'] = 'RNA_c'
ATAC_data.obs['batch'] = 'ATAC_c'

if 'cell_type' in RNA_data.obs.columns:
    if 'clusters' not in RNA_data.obs.columns:
        RNA_data.obs['clusters'] = RNA_data.obs['cell_type'].astype('category').cat.codes.astype(np.int32)
        ATAC_data.obs['clusters'] = RNA_data.obs['clusters'].astype(np.int32)  # 复制到 ATAC_data，并确保 int32
    else:
        RNA_data.obs['clusters'] = RNA_data.obs['clusters'].astype(np.int32)
        ATAC_data.obs['clusters'] = ATAC_data.obs['clusters'].astype(np.int32)
else:
    raise ValueError("No 'cell_type' in obs. Cannot generate 'clusters' without it. Please check data or run clustering algorithm.")

pca_rna_data,  rna_adata = preprocess_rna(RNA_data, n_components=50)
lsa_atac_data, atac_adata = preprocess_atac(ATAC_data, n_components=50)


data1 = sc.AnnData(rna_adata.obsm['X_pca'])
data2 = sc.AnnData(atac_adata.obsm['X_lsi'])
data1.obs_names = rna_adata.obs_names
data2.obs_names = atac_adata.obs_names
data1.obs['clusters'] = rna_adata.obs['clusters']
data1.obs['cell_type'] = rna_adata.obs['cell_type']
data2.obs['clusters'] = atac_adata.obs['clusters']
data2.obs['cell_type'] = atac_adata.obs['cell_type']


path = '../data/' + data_id
if not os.path.exists(path):
    os.makedirs(path)

data1.write_h5ad(os.path.join(path, "raw_data_rna.h5ad"))
data2.write_h5ad(os.path.join(path, "raw_data_atac.h5ad"))