import onnx
import numpy as np


def prune_weights(model: onnx.ModelProto,
                  threshold: float = 1e-3,
                  verbose: bool = False,
                  **kwargs) -> onnx.ModelProto:
    all_params, nb_zeros = 0, 0
    for initializer in model.graph.initializer:
        w_array = onnx.numpy_helper.to_array(initializer).copy()
        _mask = (np.abs(w_array) < threshold) 
        w_array[np.abs(w_array) < threshold] = 0
        new_initializer = onnx.numpy_helper.from_array(
            w_array, initializer.name)

        all_params += np.prod(w_array.shape)
        nb_zeros += np.sum(_mask)

        model.graph.initializer.remove(initializer)
        model.graph.initializer.append(new_initializer)

    if verbose:
        print(f"- Weight Pruning: {nb_zeros/all_params=:.2%}")

    return model
