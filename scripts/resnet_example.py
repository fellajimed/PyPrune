import os
import onnx
import torch
from torchvision import models
from pathlib import Path

from pyprune.quantization import onnx_quantize
from pyprune.onnx_module import ONNXModule


if __name__ == "__main__":
    project_path = Path(__file__).parents[1].resolve().absolute()
    source = project_path / 'resnet.onnx'
    dest_quant = project_path / 'resnet_quant.onnx'
    dest_pruned_quant = project_path / 'resnet_pruned_quant.onnx'

    # model
    model = models.resnet18(weights='IMAGENET1K_V1')
    dummy_input = torch.randn(1, 3, 224, 224)

    # get ONNX model from PyTorch model
    ONNXModule.export(model, dummy_input, source)

    # quantize ONNX model
    onnx_quantize(source, dest_quant)

    # get sizes
    size_source = os.stat(source).st_size / 1e3
    size_dest_quant = os.stat(dest_quant).st_size / 1e3

    # pruning
    onnx_module = ONNXModule(onnx_path=source)
    onnx_module.prune(threshold=1e-2, verbose=True)
    onnx.save(onnx_module.onnx_model, dest_pruned_quant)
    onnx_quantize(dest_pruned_quant, dest_pruned_quant)
    size_dest_pruned_quant = os.stat(dest_pruned_quant).st_size / 1e3

    print(f"* Quantization gain: {size_source/size_dest_quant=:.2%}",
          f"({size_source=:.3e}MB - {size_dest_quant=:.3e}MB)")
    print("* Pruning and Quantization gain:",
          f"{size_source/size_dest_pruned_quant=:.2%}",
          f"({size_source=:.3e}MB - {size_dest_pruned_quant=:.3e}MB)")

    # cleaning
    source.unlink(missing_ok=True)
    dest_quant.unlink(missing_ok=True)
    dest_pruned_quant.unlink(missing_ok=True)
