import logging
import sys
import contextlib
from typing import Optional, Any
import numpy

import torch
import torch.distributed._functional_collectives
import torch.func
import torch.utils._mode_utils as mode_utils
import torch.utils._python_dispatch as torch_dispatch
import torch.utils._pytree as torch_pytree

logger = logging.getLogger(__name__)


class OperatorNotFound(Exception):
  pass





class Tensor(torch.Tensor):

  @staticmethod
  def __new__(cls, meta, payload, env):
    return torch.Tensor._make_wrapper_subclass(
        cls,
        meta.shape,
        dtype=meta.dtype,
        device='meta',
        requires_grad=False,
    )

  def __init__(self, meta, payload, env: 'Environment'):
    super().__init__()
    self.payload = payload
    self._env = env

  def __str__(self):
    return "Tensor({} {})".format(str(type(self.payload)), str(self.payload))

  __repr__ = __str__

  __torch_function__ = torch._C._disabled_torch_function_impl

  @classmethod
  def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    raise NotImplementedError("Should be called inside of environment")

  @property
  def data(self):
    return self

  @property
  def device(self):
    return 'privateuse1'


class FunctionMode(torch.overrides.TorchFunctionMode):
  """Context manager that dispatches torch function calls to JAX."""

  def __init__(self, env):
     self.env = env

  def __torch_function__(self,
                         func,
                         types,
                         args=(),
                         kwargs=None) -> torch.Tensor:
    try:
      return self.env.dispatch(func, types, args, kwargs)
    except OperatorNotFound:
      pass
    return func(*args, **(kwargs or {}))


class DispatchMode(torch_dispatch.TorchDispatchMode):

  def __init__(self, env):
    self.env = env

  def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    if isinstance(func, torch._ops.OpOverloadPacket):
      with self:
        return func(*args, **kwargs)
    if func.namespace not in ('aten', '_c10d_functional', 'torchvision'):
      return func(*args, **kwargs)
    return self.env.dispatch(func, types, args, kwargs)

def _name_of_func(func):
  if hasattr(func, 'name'):
    return func.name()
  return func.__name__


# Constructors that don't take other tensor as input
TENSOR_CONSTRUCTORS = {
  torch.ones,
  torch.zeros,
  torch.empty,
  torch.empty_strided,
  torch.tensor,
  torch.arange,
  torch.eye,
  torch.randn,
  torch.rand,
  torch.randint,
  torch.full,
  torch.as_tensor,
}


class Environment:

    def __init__(self, payload_type, 
                 torch_to_payload, 
                 payload_to_torch,
                 torch_to_payload_dtype=None,
                 payload_to_torch_dtype=None):

        self._function_mode = FunctionMode(self)
        self._dispatch_mode = DispatchMode(self)

        self._ops = {}

        # Decomposition is just a torch op implemented in terms
        # of another torch op
        self._decompositions = {}

        self._manually_entered = False 
        self.enabled = False

        self._payload_type = payload_type
        self._torch_to_payload = torch_to_payload
        self._payload_to_torch = payload_to_torch

    def register_op(self, op, func):
      self._ops[op] = func

    def register_decomposition(self, op, func):
      self._ops[op] = func

    def register_default_decompositions(self):
      decomps = torch._decomp.core_aten_decompositions()
      self._decompositions.update(decomps)

    def _run_tensor_constructor_cpu(self, func, args, kwargs):
      # no tensor constructor registered, fallback to torch
      with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
        torch_tensor = func(*args, **kwargs)
        return torch_tensor



    def _handle_tensor_constructor(self, func, args, kwargs):

      device = kwargs.get('device')
      if device is None:
        device = torch.get_default_device().type

      if device == 'cpu':
        r = self._run_tensor_constructor_cpu(func, args, kwargs)
        return r

      op = self._get_op_impl(func)
      if op is not None:
        res = op(*args, **kwargs)
        meta = self._payload_to_torch(res).to('meta')
        return Tensor(meta, res, self)

      op = self._get_decomposition(func)
      if op is not None:
        with self:
          return op(*args, **kwargs)

      # no tensor constructor registered, fallback to torch
      torch_tensor = self._run_tensor_constructor_cpu(func, args, kwargs)
      payload = self._torch_to_payload(torch_tensor)
      meta = torch_tensor.to('meta')
      return Tensor(meta, payload, self)

    def _change_device(self, the_tensor, device):
      if device is None:
        return the_tensor
      if not isinstance(device, str):
        device = device.type
      old_device = the_tensor.device.type

      if old_device == 'cpu' and device == 'privateuse1':
        payload = self._torch_to_payload(the_tensor)
        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
          meta = the_tensor.to('meta')
        return Tensor(meta, payload, self)
      elif old_device == 'privateuse1' and device == 'cpu':
        return self._payload_to_torch(the_tensor.payload)

      # fallback
      with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
        return the_tensor.to(device)

      

    def _change_dtype(self, the_tensor, dtype):
      if dtype is None or the_tensor.dtype == dtype:
        return the_tensor

      if isinstance(the_tensor, Tensor):
        return self._call_op(torch.ops.aten.to, the_tensor, {'dtype': dtype})
      
      # fallback
      with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
        return the_tensor.to(dtype)


    def _to_copy(self, the_tensor, dtype, device):
      the_tensor = self._change_device(the_tensor, device)
      return self._change_dtype(the_tensor, dtype)
      
    def _torch_Tensor_to(self, args, kwargs):
      the_tensor = args[0]
      args = args[1:]
      if len(args) >= 1 and isinstance(args[0], torch.Tensor):
        dtype = args[0].dtype
        device = args[0].device
        return self._to_copy(the_tensor, dtype, device)
      device = kwargs.get('device')
      dtype = kwargs.get('dtype')
      # args like pin_memory etc that we will ignore
      args = list(filter(lambda x: not isinstance(x, bool), args))
      if len(args) >= 2:
        device, dtype, *_ = args
      elif len(args) == 1 and isinstance(args[0], torch.dtype):
        dtype = args[0]
      elif len(args) == 1:
        device = args[0]
      return self._to_copy(the_tensor, dtype, device)

    def _get_op_from_registry(self, func, registry):
      op = registry.get(func)

      if op is None and isinstance(func, torch._ops.OpOverloadPacket):
        op = registry.get(func.default)

      if op is None and isinstance(func, torch._ops.OpOverload):
        op = registry.get(func.overloadpacket)

      return op

    def _get_decomposition(self, func):
      return self._get_op_from_registry(func, self._decompositions)

    def _get_op_impl(self, func):
      return self._get_op_from_registry(func, self._ops)

    def _call_op(self, op, args, kwargs):

      def unwrap(x):
        return x.payload

      def wrap(y):
        meta = self._payload_to_torch(y).to('meta')
        return Tensor(meta, y, self)

      (args, kwargs) = torch_pytree.tree_map_only(torch.Tensor, unwrap, (args, kwargs))
      res = op(*args, **kwargs)
      return torch_pytree.tree_map_only(self._payload_type, wrap, res)


    def dispatch(self, func, types, args, kwargs):

      kwargs = kwargs or {}
      if func in TENSOR_CONSTRUCTORS:
        return self._handle_tensor_constructor(func, args, kwargs)
      if func in (torch.Tensor.to, torch.ops.aten.lift_fresh.default ,torch.ops.aten._to_copy, torch.ops.aten._to_copy.default):
        return self._torch_Tensor_to(args, kwargs)

      # If the func doesn't act on Tensor, and is not a tensor constructor,
      # We should skip and let torch handle it.
      
      tensor_args = [t for t in torch_pytree.tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
      if tensor_args and all(not isinstance(t, Tensor) for t in tensor_args):
        return func(*args, **kwargs)

      op = self._get_op_impl(func)

      if op is not None:
        return self._call_op(op, args, kwargs)

      op = self._get_decomposition(func)

      if op is None:
        raise OperatorNotFound(
          f'Operator with name {_name_of_func(func)} has no lowering')

      args, kwargs = torch_pytree.tree_map_only(
          torch.distributed._functional_collectives.AsyncCollectiveTensor,
          torch.distributed._functional_collectives.wait_tensor,
          (args, kwargs))

      with self._function_mode, self._dispatch_mode:
        return op(*args, **kwargs)

    def enable_torch_modes(self):
      self._dispatch_mode.__enter__()
      self._function_mode.__enter__()
      self.enabled = True
    
    def disable_torch_modes(self, *exc):
      if not exc:
        exc = (None, None, None)
      self._function_mode.__exit__(*exc)
      self._dispatch_mode.__exit__(*exc)
      self.enabled = False
