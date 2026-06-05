# SCARF: A Single Cell ATAC-seq and RNA-seq Foundation Model

SCARF is a foundation model for single-cell RNA-seq, ATAC-seq, and paired
multiome data. It provides pretrained weights, preprocessing utilities,
embedding inference notebooks, and downstream analysis examples.

## System Requirements

- Operating system: Linux, Ubuntu 20.04 or later recommended.
- Python: 3.12.3.
- GPU: NVIDIA GPU with CUDA 11.8 recommended for model inference.
- CPU-only inference is supported for small examples, with longer runtime.
- Memory: depends on dataset size and modality; use the preprocessing scripts
  with local scratch storage for large AnnData objects.

## Installation

### Conda

```bash
git clone https://github.com/JiekaiLab/scarf.git
cd scarf
CONDA_CHANNEL_PRIORITY=flexible conda env create -f environment.yml
conda activate scarf
pip install -e .
```



### Docker

If you use the prepared Docker image, mount this repository and the data
directory into the container:

```bash
docker run --gpus all --rm -it \
  -v /path/to/scarf-main:/workspace/scarf \
  -v /path/to/data:/workspace/data \
  -w /workspace/scarf \
  scarf:latest bash
```

Inside the container:

```bash
pip install -e .
jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
```

More Docker notes are available in [docker/README.md](docker/README.md).

## Download Required Files

Run [downstream_tasks/download_data.ipynb](downstream_tasks/download_data.ipynb)
or download the archives manually from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17205044.svg)](https://doi.org/10.5281/zenodo.17205044)

Place the extracted files under:

- `weights/`: pretrained model configuration and checkpoint shards.
- `prior_data/`: token dictionaries and prior statistics.
- `data/`: optional demo or raw input data.

See [prior_data/README.md](prior_data/README.md) for the role of each prior
file, including the pickle files required by preprocessing.

## Quick Start

### 1. Preprocess Data

Use the standalone preprocessing entrypoint:

```bash
python scripts/preprocessing/scM_convert.py
```

The script converts scRNA, scATAC, or scMultiome AnnData inputs into a
HuggingFace `Dataset` compatible with SCARF embedding inference. Configure the
input paths, species, modality, and output directory near the top of
`scripts/preprocessing/scM_convert.py`.

Additional notes are in
[scripts/preprocessing/README.md](scripts/preprocessing/README.md).

### 2. Run Embedding Inference

Open and run:

- [downstream_tasks/embedding.ipynb](downstream_tasks/embedding.ipynb)

The notebook loads a preprocessed dataset with `datasets.load_from_disk()` and
generates RNA and/or ATAC cell embeddings.

### 3. Run Downstream Tasks

Example notebooks are provided under `downstream_tasks/`:

- [CellType_prediction.ipynb](downstream_tasks/CellType_prediction.ipynb):
  cell type prediction from embeddings.
- [Cell-matching.ipynb](downstream_tasks/Cell-matching.ipynb): cross-modality
  cell matching.
- [RNA-Inference.ipynb](downstream_tasks/RNA-Inference.ipynb): RNA inference
  from ATAC-derived representations.

Additional application-oriented analysis scripts are available in
[downstream_tasks/application_analysis](downstream_tasks/application_analysis).

## Repository Structure

```text
SCARF/
|-- data/                 # Demo or user-provided data, not committed
|-- docker/               # Docker usage notes
|-- downstream_tasks/     # Notebooks and downstream analysis scripts
|-- prior_data/           # Token dictionaries and prior statistics
|-- scarf/                # Model and utility package
|-- scripts/              # Preprocessing scripts
|-- weights/              # Pretrained model files, downloaded separately
|-- environment.yml       # Conda environment
`-- pyproject.toml        # Editable package installation
```

## Notes on Large Files

Large checkpoints, raw AnnData files, processed datasets, embeddings, and most
pickle prior files are intentionally excluded from version control. They should
be downloaded from the project archive or generated locally.

## License

This project is released under the GNU General Public License v3.0. See
[LICENSE](LICENSE) for details.

## Citation

If you use SCARF in your research, please cite:

```bibtex
@misc{SCARF2025,
  title   = {SCARF: A Single Cell ATAC-seq and RNA-seq Foundation Model},
  author  = {Guole Liu#, Tianyu Wang#, Yingying Zhao#, Quanyou Cai#, Xiaotao Wang#, Ziyi Wen, Yaofeng Wang, Lihui Lin*, Yongbing Zhao*, Ge Yang*, Jiekai Chen*},
  year    = {2025},
  url     = {https://github.com/JiekaiLab/scarf},
  doi     = {https://doi.org/10.1101/2025.04.07.647689}
}
```
