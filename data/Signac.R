
library(Signac)
library(sp)
library(SeuratObject)
library(Seurat)
library(reticulate)
library(sceasy)
library(EnsDb.Mmusculus.v79)  # 使用小鼠基因组注释（假设v79版本，根据需要调整）
library(zellkonverter)
library(rhdf5)
library(anndata)
# 读取ATAC数据
sce <- readH5AD("E:/data/raw_data/GSE126074_AdBrainCortex/ATAC/ATAC.h5ad")

data.atac <- sce@assays@data@listData[["X"]]

# 过滤ATAC峰，仅保留标准染色体
grange.counts <- StringToGRanges(rownames(data.atac), sep = c(":", "-"))
grange.use <- seqnames(grange.counts) %in% standardChromosomes(grange.counts)
data.atac <- data.atac[as.vector(grange.use), ]

# 获取小鼠基因组注释
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Mmusculus.v79)
seqlevelsStyle(annotations) <- 'UCSC'
genome(annotations) <- "mm10"  # 小鼠基因组版本mm10

# 指定fragments文件路径（替换为实际路径；GSE126074数据集可能有对应的fragments文件，如GSE126074_AdBrainCortex_SNAREseq_chromatin.fragments.tsv.gz）
frag.file <- "E:/data/raw_data/GSE126074_AdBrainCortex/ATAC/fragments.sort.bed.gz"  # 假设路径，请确认并替换
# 不使用fragments文件
#frag.file <- NULL
#cat("未使用fragments文件，依赖峰计数矩阵生成基因活性矩阵。\n")

# 创建ChromatinAssay对象
chrom_assay <- CreateChromatinAssay(
  counts = data.atac,
  sep = c(":", "-"),
  genome = 'mm10',
  fragments = frag.file,
  min.cells = 10,
  annotation = annotations
)

# 创建Seurat对象
AdBrainCortex <- CreateSeuratObject(
  counts = chrom_assay,
  assay = "ATAC"
)

# 设置默认assay并计算基因活性
DefaultAssay(AdBrainCortex) <- "ATAC"
gene.activities <- GeneActivity(object = AdBrainCortex)

# 创建基因活性Seurat对象并转换为AnnData格式
scgeneactivity <- CreateSeuratObject(counts = gene.activities, assay = "RNA")
sceasy::convertFormat(scgeneactivity, from = "seurat", to = "anndata", outFile = 'E:/data/raw_data/GSE126074_AdBrainCortex/ATAC/ATAC_Signac.h5ad')
