#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import random
import pandas as pd
from load_result import *
from evaluation import evaluate
import warnings
warnings.filterwarnings("ignore")
random.seed(666)

dataset_name = "Tea_PBMC"
dataset_type = 'RNA_ATAC'
GAM_name = 'ArchR'
paired = True

path = "E:/experiment/scGSI/"
dataset_dir = path + 'data/'
result_dir = path + "results/" + dataset_name
eva_dir = path + "eva/" + dataset_name
vis_dir = path + 'vis/' + dataset_name

if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)
if not os.path.exists(eva_dir):
    os.makedirs(eva_dir)

# adata = anndata.read_h5ad(result_dir + 'adata.h5ad')
# adata_gasm = anndata.read_h5ad(result_dir + 'adata_gasm.h5ad')
# annotations = np.load(result_dir + "/annotations.npz")
# anno_rna = annotations['anno_rna']
# anno_other = annotations['anno_other']
# anno_rna_gasm = annotations['anno_rna_gasm']
# anno_other_gasm = annotations['anno_other_gasm']

methods = ['raw_data', 'JointMDS', 'MMDMA', 'Pamona', 'SCOT', 'UnionCom', 'scTopoGAN', 'scMGCL', 'scPairing', 'scGSI']

adata, anno_rna, anno_atac, clu_rna, clu_atac = load_result(dataset_name, dataset_dir, result_dir, methods)

labels = [clu_rna, clu_atac]
adata.obs_names_make_unique()  # 确保观察名称唯一
adata.write(result_dir+"/adata.h5ad", compression='gzip')
np.savez_compressed(result_dir+"/annotations.npz", anno_rna=anno_rna, anno_atac=anno_atac)

eva_metrics, mix_and_bio = evaluate(adata, anno_rna, anno_atac, 10, labels, eva_dir)
eva_metrics.T.to_csv(eva_dir + '/eva_metrics_' + dataset_name + '.csv', index=True, float_format='%.3f')
mix_and_bio.T.to_csv(eva_dir + '/mix_and_bio_metrics_' + dataset_name + '.csv', index=True, float_format='%.3f')

