import math
import torch

class StdAttention(torch.nn.Module):

  def attention_score(self, q, k, v):
    a = torch.einsum('...id,...jd->...ij', q, k)
    a = torch.exp(a / math.sqrt(float(q.shape[-1])))
    d_inv = 1.0 / a.sum(dim=-1, keepdims=True)
    return d_inv * a

  def forward(self, q, k, v):
    a = torch.einsum('...id,...jd->...ij', q, k)
    a = torch.exp(a / math.sqrt(float(q.shape[-1])))
    d_inv = 1.0 / a.sum(dim=-1, keepdims=True)
    a_hat = d_inv * a
    return a_hat @ v


class FastAttention(torch.nn.Module):
  def __init__(self, qk_dim, kernel, sample_dim=None, device=None):
    super(FastAttention, self).__init__()
    self.qk_dim = qk_dim
    self.kernel = kernel
    self.sample_dim = sample_dim
    self.device = device
    self.set_projection_matrix()

  def set_projection_matrix(self):
    projection_matrix = torch.tensor(
        self.kernel.w(self.sample_dim, self.qk_dim),
        dtype=torch.float32, device=self.device)
    self.register_buffer('projection_matrix', projection_matrix)

  def kernelize(self, data):
    data_norm = (data.shape[-1] ** (-0.25)) * data
    ratio = (self.projection_matrix.shape[0] ** -0.5)
    data_dash = torch.einsum('...id,rd->...ir', data_norm, self.projection_matrix)
    return ratio * self.kernel.h(data_norm) * self.kernel.f(data_dash)

  def attention_score(self, q, k, v):
    q_hat = self.kernelize(q)
    k_hat = self.kernelize(k)
    a_hat = q_hat @ k_hat.T
    d_inv = 1.0 / a_hat.sum(dim=-1, keepdims=True)
    return d_inv * a_hat

  def forward(self, q, k, v):
    q_hat = self.kernelize(q)
    k_hat = self.kernelize(k)
    z = k_hat.T @ v
    w = q_hat @ z
    d_inv = 1.0 / ((q_hat * k_hat.sum(axis=0)).sum(dim=1, keepdims=True))
    return d_inv * w

