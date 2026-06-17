# 阶段1: 基础系统 (Ubuntu 22.04 匹配 CUDA 11.8 官方支持)
FROM ubuntu:22.04

# 阶段2: 安装系统依赖和 CUDA 11.8 工具包
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    bzip2 \
    build-essential \
    curl \
    git \
    libxml2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 阶段3: 安装 CUDA Toolkit 11.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-toolkit-11-8 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/cuda-11.8/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64" \
    CUDA_HOME="/usr/local/cuda-11.8"

# 阶段4: 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH="/opt/conda/bin:${PATH}"
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda init bash

# 阶段5: 创建并配置 scarf 虚拟环境（内嵌 environment.yml）
WORKDIR /workspace/scarf

RUN cat > environment.yml <<'EOF'
name: scarf
channels:
  - conda-forge
  - nvidia
dependencies:
  - python=3.12.3
  - cudatoolkit=11.8
  - nvidia::cuda-nvcc=11.8.89
  - pip
  - pip:
      - torch==2.3.1 --index-url https://download.pytorch.org/whl/cu118
      - torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
      - torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
      - triton==2.3.1
      - accelerate==1.0.1
      - transformers[torch]==4.46.3
      - datasets==2.21.0
      - geomloss==0.2.6
      - numpy==1.26.4
      - pandas==2.2.3
      - scipy==1.14.1
      - h5py==3.11.0
      - pyarrow==17.0.0
      - tqdm==4.66.5
      - matplotlib==3.9.2
      - seaborn==0.13.2
      - umap-learn==0.5.6
      - scikit-learn==1.5.2
      - anndata==0.9.0
      - scanpy==1.11.0
      - jupyterlab==4.2.5
      - notebook==7.2.2
      - ipykernel==6.29.5
      - ipywidgets==8.1.5
      - https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
      - https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.3cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
      - nmslib
EOF

RUN conda env create -f environment.yml -n scarf
RUN echo "conda activate scarf" >> ~/.bashrc
ENV PATH="/opt/conda/envs/scarf/bin:${PATH}"

# 验证环境并确保关键包存在，若缺失则安装，并明确指定 scib 版本为 1.1.5
RUN /bin/bash -c "conda activate scarf && pip list | grep -E 'jupyter|datasets|geomloss|pysankey|POT|scib' || (echo 'Installing missing packages...' && pip install jupyter datasets geomloss pysankey POT scib==1.1.5)" && \
    echo "Core packages verified."

# 阶段6: 复制项目代码
COPY . .

WORKDIR /workspace/scarf
CMD ["/bin/bash", "-c", "conda activate scarf && exec /bin/bash"]