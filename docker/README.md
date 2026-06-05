# Docker Usage

Use this file to document and run the prepared SCARF Docker image. Replace the
image name below with the image tag used in your local registry.

## Start an Interactive Container

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

## Run Preprocessing

```bash
python scripts/preprocessing/scM_convert.py
```

## Run Without GPU

```bash
docker run --rm -it \
  -v /path/to/scarf-main:/workspace/scarf \
  -v /path/to/data:/workspace/data \
  -w /workspace/scarf \
  scarf:latest bash
```

## Expected Mounted Files

- Repository: mounted at `/workspace/scarf`.
- Raw data and generated datasets: mounted separately, for example
  `/workspace/data`.
- Pretrained model weights: available under `/workspace/scarf/weights`.
- Prior data files: available under `/workspace/scarf/prior_data`.
