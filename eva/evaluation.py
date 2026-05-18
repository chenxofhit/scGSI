
from eval import *


def evaluate(adata, anno_rna, anno_atac, n_neighbors, labels, eva_dir):

    eva_metrics = {}
    mix_and_bio_metrics = {}
    obsm_names = adata.obsm_keys()
    n_clusters = len(np.unique(adata.obs['cluster']))
    for i in range(len(obsm_names)):
        method = obsm_names[i]
        z_rna = adata.obsm[method][0:len(anno_rna), :]
        z_atac = adata.obsm[method][len(anno_rna):, :]
        evals, mix_and_bio = evaluate_alignment(z_rna, z_atac, labels, n_clusters, n_neighbors, method, eva_dir)
        eva_metrics[method] = evals
        mix_and_bio_metrics[method] = mix_and_bio
        print("evaluate finished :", method)
    eva_metrics = pd.DataFrame(eva_metrics)
    eva_metrics.index = ['FOSCTTM', 'LTA', 'ARI', 'NMI', 'AMI', 'OMI', 'BCI', 'Accuracy']
    mix_and_bio_metrics = pd.DataFrame(mix_and_bio_metrics)
    mix_and_bio_metrics.index = ['NOC', 'GC', 'SAS', 'ASW', 'PS', 'MAP']

    return eva_metrics, mix_and_bio_metrics
