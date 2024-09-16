## Env

```bash
conda create -n prune_env python=3.11
python -m pip install -r requirements.txt
```


## Install library

```bash
python -m pip install -e .
```

## Run main

```bash
pyprune --source [onnx_source] --dest [onnx_dest]
```
