import time
import torch
import kernel
import unittest
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fast_attention import StdAttention, FastAttention
from ortho import gaussian_orthogonal_random_matrix


class TestFastAttentionFigure(unittest.TestCase):

  def test_case(self):
    np.random.seed(1231)
    l, d = 5, 64

    q = torch.tensor(np.random.randn(l, d), dtype=torch.float32)
    k = torch.tensor(np.random.randn(l, d), dtype=torch.float32)
    v = torch.tensor(np.random.randn(l, d), dtype=torch.float32)

    attention_layer = StdAttention()
    std_attention_score = attention_layer.attention_score(q, k, v).numpy()
    np.testing.assert_array_equal(std_attention_score.shape, (l, l))
    np.testing.assert_array_almost_equal(std_attention_score.sum(1), [1.0]*l)

    attention_layer = FastAttention(d, kernel.SMplusKernel(), 128)
    fast_attention_score = attention_layer.attention_score(q, k, v).numpy()
    np.testing.assert_array_equal(fast_attention_score.shape, (l, l))
    np.testing.assert_array_almost_equal(fast_attention_score.sum(1), [1.0]*l)
    
    plt.subplot(1, 2, 1)
    a = sns.heatmap(std_attention_score, cmap='Blues', vmin=0.0, vmax=1.0)
    plt.subplot(1, 2, 2)
    b = sns.heatmap(fast_attention_score, cmap='Blues', vmin=0.0, vmax=1.0)
    plt.savefig('attention_score.png')


class TestAttentionSpeed(unittest.TestCase):

  def test_case(self):
    d = 128
    ls = [2**i for i in (range(3, 14))]
    device = torch.device('cpu')

    for l in ls:
      q = torch.randn(l, d, device=device)
      k = torch.randn(l, d, device=device)
      v = torch.randn(l, d, device=device)

      start_time = time.time()
      attention_layer = StdAttention()
      for _ in range(10):
        std_res = attention_layer(q, k, v)
        np.testing.assert_array_equal(std_res.shape, (l, d))
      std_time = time.time() - start_time

      start_time = time.time()
      attention_layer = FastAttention(d, kernel.SMplusKernel(), 128, device=device)
      for _ in range(10):
        fast_res = attention_layer(q, k, v)
        np.testing.assert_array_equal(fast_res.shape, (l, d))
      fast_time = time.time() - start_time

      mse = torch.nn.functional.mse_loss(std_res, fast_res)
      print('%4d:\tstd: %f\tfast: %f\tmse: %.2f' 
          % (l, std_time, fast_time, mse))


class TestOrthoRandomPrint(unittest.TestCase):

  def test_case(self):
    np.random.seed(1231)
    mat = gaussian_orthogonal_random_matrix(3, 5, scaling=0)
    # print(mat @ mat.T)
    mat = gaussian_orthogonal_random_matrix(5, 3, scaling=1)
    # print(mat @ mat.T)
    mat = gaussian_orthogonal_random_matrix(4, 4, scaling=0)
    # print(mat @ mat.T)


if __name__ == "__main__":
  unittest.main()
