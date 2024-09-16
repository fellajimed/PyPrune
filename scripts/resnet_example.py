import os
import torch
from torchvision import models
from pathlib import Path

from pyprune.quantization import onnx_quantize
from pyprune.onnx_module import ONNXModule


if __name__ == "__main__":
    project_path = Path(__file__).parents[1].resolve().absolute()
    source = project_path / 'resnet.onnx'
    dest = project_path / 'resnet_quant.onnx'

    # model
    model = models.resnet18(weights='IMAGENET1K_V1')
    dummy_input = torch.randn(1, 3, 224, 224)

    # get ONNX model from PyTorch model
    ONNXModule.export(model, dummy_input, source)

    # quantize ONNX model
    onnx_quantize(source, dest)

    size_source = os.stat(source).st_size
    size_dest = os.stat(dest).st_size

    print(f"* Quantization gain: {size_source/size_dest=:.2%}",
          f"({size_source=:.3e}B - {size_dest=:.3e}B)")

    # cleaning
    source.unlink(missing_ok=True)
    dest.unlink(missing_ok=True)
