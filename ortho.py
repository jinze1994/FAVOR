import numpy as np


def _gaussian_orthogonal_chunk(cols: int, qr_uniform_q=False):
  unstructured_block = np.random.randn(cols, cols)
  q, r = np.linalg.qr(unstructured_block)

  # proposed by @Parskatt
  # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
  if qr_uniform_q:
    q *= r.diagonal().sign()
  return q.T


def gaussian_orthogonal_random_matrix(
    rows: int, cols: int, scaling=0, qr_uniform_q=False):
    n_blocks = int(rows / cols)
    block_list = [
        _gaussian_orthogonal_chunk(cols, qr_uniform_q=qr_uniform_q)
        for _ in range(n_blocks) ]

    remaining_rows = rows - n_blocks * cols
    if remaining_rows > 0:
      q = _gaussian_orthogonal_chunk(cols, qr_uniform_q=qr_uniform_q)
      block_list.append(q[:remaining_rows])

    final_matrix = np.concatenate(block_list, axis=0)

    if scaling == 0:
        multiplier = np.linalg.norm(
            np.random.randn(rows, cols), axis=1, keepdims=True)
    elif scaling == 1:
        multiplier = np.sqrt(float(cols))
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return multiplier * final_matrix
