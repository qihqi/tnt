# TNT's Not a Transpiler

This is a short demo of using `__torch_dispatch__` and `__torch_function__`
modes to accomplish the registration of an Apple MLX backend in PyTorch using Python only.


In this setup, every torch op is actually executed using MLX.

This idea is described in https://dev-discuss.pytorch.org/t/embrace-tensor-subclass-as-a-python-device-registration-api/2771 and 
inspired from https://github.com/albanD/subclass_zoo/blob/main/new_device.py

[torchax](https://github.com/pytorch/xla/tree/master/torchax) Also uses the 
same mechanism but using Jax as the backend.

In this example, we only registers the bare-minimum to be able to run the llama model.
I used a much more smaller scale of the model, however all the operators needed to 
run the full version (say 8B) is completed here.

# Rough steps:

1. Define your payload type (here it is `mlx.array`).
2. Tell the Environment how to transform a `torch.Tensor` to your payload and vice-versa
3. Register the ATen ops that needs to run your model. Each registration is to implement
   the logic of an ATen op using MLX ops. LLMs can help a lot here.

After all done:
To run it:

```bash
python torch_mlx.py
```
