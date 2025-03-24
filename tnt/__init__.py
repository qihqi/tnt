import torch.utils._python_dispatch as torch_dispatch


def print_missing_ops(func, args, kwargs):

  class PrintMode(torch_dispatch.TorchDispatchMode):

    def __init__(self):
      self._ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
      message = f'torch_dispatch {func}'
      print(message)
      self._ops.add(func)
      kwargs = kwargs or {}
      return func(*args, **kwargs)

  mode = PrintMode()
  with mode:
    print(func(*args, **kwargs))

  for op in mode._ops:
    print(
  f'''
  @register_op({op})
  def {str(op).replace(".", "_")}():
      pass
  '''
    )