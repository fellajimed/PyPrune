import torch
import onnx
import onnxruntime
from onnx2pytorch import ConvertModel
from os import PathLike
from pathlib import Path
import numpy as np

from .prune import prune_weights


class ONNXModule:
    def __init__(self, *, 
                 onnx_model: onnx.ModelProto | None = None,
                 onnx_path: str | PathLike | Path | None = None,
                 ) -> None:
        if onnx_model is not None:
            self.onnx_model = onnx_model
            onnx_model_serialized = onnx_model.SerializeToString()
            self.session = onnxruntime.InferenceSession(onnx_model_serialized)
        elif onnx_path is not None:
            self.onnx_model = onnx.load(onnx_path)
            self.session = onnxruntime.InferenceSession(onnx_path)
        else:
            raise ValueError('not a valid path and not a valid onnx model')

        onnx.checker.check_model(self.onnx_model)


    def forward(self,
                inputs: np.ndarray | torch.Tensor
                ) -> np.ndarray:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.numpy()

        assert isinstance(inputs, np.ndarray)

        # Get the input name for the ONNX model
        input_name = self.session.get_inputs()[0].name

        # Perform inference
        return self.session.run(None, {input_name: inputs})[0]


    def prune(self, strategy: str = "weights", **kwargs) -> None:
        if strategy.lower() == "weights":
            prune_weights(self.onnx_model, **kwargs)


    def to_pytorch(self,
                   onnx_model: onnx.ModelProto | None = None
                   ) -> torch.nn.Module:
        # FIXME: requires batch_size=1
        if onnx_model is None:
            onnx_model = self.onnx_model

        return ConvertModel(onnx_model)


    @staticmethod
    def export(model: torch.nn.Module,
               dummy_input: torch.Tensor,
               fname: str | PathLike | Path,
               ) -> None:
        torch.onnx.export(model, dummy_input, fname,
                          export_params=True, do_constant_folding=True,
                          input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

if __name__ == "__main__":
    from torch import nn


    class TestModel(nn.Module):
        """
        A simple torch model
        """
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(20, 5),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layers(x)


    # create a PyTorch model
    model = TestModel()
    # set it to eval mode
    model.eval()

    # export the PyTorch model as ONNX model
    dest = Path(__file__).parents[2].resolve().absolute() / 'model.onnx'
    ONNXModule.export(model, torch.randn(1, 10), dest)

    # init ONNX Module
    onnx_module = ONNXModule(onnx_path=dest)
    
    inputs = torch.randn(7, 10)
    onnx_outputs = onnx_module.forward(inputs)

    # tests
    assert onnx_outputs.shape == (7, 5)
    assert torch.allclose(torch.from_numpy(onnx_outputs), model(inputs))

    # get PyTorch model
    model_from_onnx = onnx_module.to_pytorch()
    model_from_onnx.eval()
    inputs = torch.randn(1, 10)

    assert torch.allclose(model(inputs), model_from_onnx(inputs))

    print('Success!!!')

    # delete onnx file
    dest.unlink(missing_ok=True)
