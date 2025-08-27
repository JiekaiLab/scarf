# SCARF: A Single Cell ATAC-seq and RNA-seq Foundation Model

SCARF is a large-scale foundation model designed for **single-cell ATAC-seq and RNA-seq**.  
It provides pretrained weights, preprocessing pipelines, and tutorials to accelerate downstream biological discovery.

---

## 🚀 System Requirements

- **Operating system**: Linux (Ubuntu 20.04+), macOS, Windows (via WSL2)
- **Python version**: == 3.12.3
- **Dependencies**:
  - PyTorch >= 2.3.1
  - Scanpy >= 1.11.0
  - Anndata >= 0.9
  - scikit-learn == 1.5.2
  - transformers==4.46.3
  - numpy, pandas, matplotlib, seaborn, jupyter
- **Hardware**:
  - CPU: x86_64 architecture (tested on Intel i9 and AMD EPYC)
  - GPU (recommended): NVIDIA GPU with CUDA >= 11.8 (tested on A800)
  - Minimum RAM: 40 GB

---

## ⚙️ Installation Guide

### 1. Clone the repository
```bash
git clone https://github.com/JiekaiLab/scarf.git
cd scarf
```
### 2. Create conda environment
```bash
conda create -n scarf python=3.12.3 -y
conda activate scarf
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Install GPU-enabled PyTorch
Please refer to the official PyTorch installation guide(https://pytorch.org/get-started/locally/).

## 📊 Quick start

We provide example datasets and pretrained models for quick testing. 
### Expected runtime on a normal desktop (40GB RAM, no GPU): ~2–3 minutes

### Run SCARF on your own data

1. Preprocess your single-cell data(待修改):
```bash
   python preprocess/preprocess.py --input data/raw.h5ad --output ./output
```

2. Run inference(待修改):
```bash
   python get_embedding.py --input data/processed.h5ad --weights weights/
```

## 🎯Downstream Tasks

We provide ready-to-use Jupyter notebooks demonstrating how to apply **SCARF** to different downstream tasks:
- **Cell type prediction** ([CellType_prediction.ipynb](./downstream_tasks/CellType_prediction.ipynb))  
  Predicts cell type labels from multi-omic embeddings.

- **Cell Matching** ([Cell-matching.ipynb](./downstream_tasks/Cell-matching.ipynb))  
  Aligns and matches cells across modalities (scRNA-seq and scATAC-seq).

- **Cell RNA-Inference** ([RNA-Inference.ipynb](./downstream_tasks/RNA-Inference.ipynb))  
  Predicts gene expression of cells through scATAC-seq data.

## 📂 Repository Structure

```
SCARF/
├── data/                 # data for demo
├── downstream_tasks/     # Jupyter notebooks for demo and usage
├── scarf/                # model file
├── prior_data/           # Token dictionaries and metadata
├── scripts/              # Preprocessing and inference scripts
├── weights/              # Pretrained model weights (download from Zenodo)
└── requirements.txt      # Dependencies
```

---

## 📜 License

This project is released under the **GNU General Public License v3.0**.  
See [LICENSE](./LICENSE) for details.

---

## 🔗 Links

* GitHub Repository: [JiekaiLab/scarf](https://github.com/JiekaiLab/scarf)
* Pretrained weights & large files: [Zenodo DOI](https://doi.org/10.5281/zenodo.16956913)

---

## 📖 Citation

If you use SCARF in your research, please cite:

```bibtex
@misc{SCARF2025,
  title   = {SCARF: A Single Cell ATAC-seq and RNA-seq Foundation Model},
  author  = {Guole Liu#,Tianyu Wang#,Yingying Zhao#,Quanyou Cai#,Xiaotao Wang#,Ziyi Wen,Lihui Lin*, Yongbing Zhao*, Ge Yang*,Jiekai Chen*},
  year    = {2025},
  url     = {https://github.com/JiekaiLab/scarf},
  doi     = {https://doi.org/10.1101/2025.04.07.647689}
}
```

---

