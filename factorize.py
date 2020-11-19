import torch
import numpy as np

class GraphDataset(torch.utils.data.Dataset):

  def __init__(self, n_vertex, edge_list):
    super(GraphDataset, self).__init__()
    self.n_vertex = n_vertex
    self.edge_list = edge_list

  def __len__(self):
    return self.edge_list.shape[0]

  def __getitem__(self, idx):
    source, pos = self.edge_list[idx]
    neg = np.random.randint(0, self.n_vertex)
    return source, pos, neg


class MatrixFactorization(torch.nn.Module):
  def __init__(self, n_vertex, out_dim):
    super(MatrixFactorization, self).__init__()
    self.u_emb = torch.nn.Embedding(n_vertex, out_dim)
    self.v_emb = torch.nn.Embedding(n_vertex, out_dim)

  def forward(self, i, j):
    u = self.u_emb(i)
    v = self.v_emb(j)
    return (u * v).sum(-1)

  def get_embedding(self):
    return self.u_emb.weight.cpu().detach().numpy(), \
        self.v_emb.weight.cpu().detach().numpy()


def factorize(n_vertex, edge_list, out_dim=128,
    max_epoch=100, batch_size=256, learning_rate=0.01,
    save_path='emb.npz'):

  device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
  assert edge_list.ndim == 2 and edge_list.shape[1] == 2
  n_edge = edge_list.shape[0]

  dataset = GraphDataset(n_vertex, edge_list)
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=batch_size, shuffle=True, num_workers=1)
  model = MatrixFactorization(n_vertex, out_dim).to(device)
  bce_loss = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  step = 0
  for epoch in range(max_epoch):
    for i_batch, sample_batched in enumerate(dataloader):
      source, pos, neg = map(lambda x: x.to(device), sample_batched)
      l1, l2 = model(source, pos), model(source, neg)
      logits = torch.cat([l1, l2])
      labels = torch.cat([torch.ones_like(l1), torch.zeros_like(l2)])
      loss = bce_loss(logits, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if step % 10 == 0:
        print('Epoch: %d\tStep: %d\tLoss: %.4f' % (
          epoch, step, loss.item()))
      step += 1

  u_emb, v_emb = model.get_embedding()
  if save_path is not None:
    np.savez(save_path, u_emb=u_emb, v_emb=v_emb)
  return u_emb, v_emb


def read_cora_edge_list():
  with open('cora/cora.content') as f:
    m, n = {}, 0
    for i, line in enumerate(f):
      m[int(line.split('\t', 2)[0])] = i
      n += 1
    assert n == len(m)

  with open('cora/cora.cites') as f:
    edge_list = []
    for line in f:
      edge_list.append(tuple(map(
        lambda x: m[int(x)],line.strip().split())))
  return n, np.array(edge_list, dtype=np.int64)


if __name__ == '__main__':
  np.random.seed(1231)
  np.set_printoptions(precision=3, suppress=True)

  n, edge_list = read_cora_edge_list()
  print(n, edge_list.shape)
  a = np.zeros((n, n), dtype=np.float32)
  a[edge_list[:, 0], edge_list[:, 1]] = 1.0
  print(a, a.sum())

  u, v = factorize(n, edge_list)
  a_hat = 1 / (1 + np.exp(- u @ v.T))

  print(a_hat, a_hat.sum())
