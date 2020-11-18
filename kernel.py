import torch
import numpy as np
from ortho import gaussian_orthogonal_random_matrix


class BaseKernel:
  def __init__(self, is_ortho=True):
    self.is_ortho = is_ortho

  def h(self, x):
    raise NotImplemented

  def f(self, x):
    raise NotImplemented

  def w(self, sample_dim, qk_dim):
    if self.is_ortho:
      return gaussian_orthogonal_random_matrix(sample_dim, qk_dim)
    else:
      return np.random.randn(sample_dim, qk_dim)


class GaussKernel(BaseKernel):
  def __init__(self, is_ortho=True):
    super(GaussKernel, self).__init__(is_ortho)

  def h(self, x):
    return 1

  def f(self, x):
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class SMtrigKernel(BaseKernel):
  def __init__(self, is_ortho=True):
    super(SMtrigKernel, self).__init__(is_ortho)

  def h(self, x):
    return torch.exp(torch.square(x).sum(dim=-1, keepdims=True) / 2)

  def f(self, x):
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class SMplusKernel(BaseKernel):
  def __init__(self, is_ortho=True):
    super(SMplusKernel, self).__init__(is_ortho)

  def h(self, x):
    return torch.exp(- torch.square(x).sum(dim=-1, keepdims=True) / 2)

  def f(self, x):
    return torch.exp(x)


class SMhypKernel(BaseKernel):
  def __init__(self, is_ortho=True):
    super(SMhypKernel, self).__init__(is_ortho)

  def h(self, x):
    ratio = 1.0 / torch.sqrt(2.0)
    return ratio * torch.exp(- torch.square(x).sum(dim=-1, keepdims=True) / 2)

  def f(self, x):
    return torch.exp(x)
