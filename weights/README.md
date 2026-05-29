# Pretrained Weights

The pretrained model files are archived on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17205044.svg)](https://doi.org/10.5281/zenodo.17205044)

Download and extract the model archive into this folder.

## Required Files

The inference utilities expect the following files to live in the same
`weights/` directory:

- `config.json`
- `pytorch_model.bin.index.json`
- `pytorch_model-00001-of-00002.bin`
- `pytorch_model-00002-of-00002.bin`

The `.bin` checkpoint shards are large and are intentionally ignored by git.
They must be downloaded from the project archive before running embedding
inference.
