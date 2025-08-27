#!/bin/bash

conda install cudatoolkit==11.8 -c nvidia
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

pip install triton==2.3.1
pip install transformers[torch]==4.46.3
pip install scikit-learn==1.5.2
pip install Anndata==0.9.0
pip install Scanpy==1.11.0


conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
