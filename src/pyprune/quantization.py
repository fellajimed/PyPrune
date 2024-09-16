from os import PathLike
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic


def onnx_quantize(source: str | PathLike | Path,
                  dest: str | PathLike | Path,
                  ) -> None:
    quantize_dynamic(source, dest)
