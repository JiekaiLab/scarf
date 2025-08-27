## environment setup
### Prerequisites
NVIDIA GPU with CUDA 11.8 compatibility

Conda package manager (Miniconda or Anaconda)

### Installation Methods
#### Method 1: Using environment.yml
conda env create -f environment.yml

#### Method 2: Manual Installation
##### Create and activate the base environment:

conda create -p scarf-env python=3.12.3 -c conda-forge
conda activate scarf-env
##### Install CUDA toolkit:

conda install -c nvidia/label/cuda-11.8.0 cudatoolkit=11.8.0 cuda-nvcc=11.8.89
##### Install PyTorch with CUDA 11.8 support:

pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 --index-url https://download.pytorch.org/whl/cu118
##### Install core packages:

pip install scanpy==1.11.0 scib==1.1.7 scikit-learn==1.5.2 transformers==4.46.3
