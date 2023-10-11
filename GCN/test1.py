import numpy as np
import torch
from torch import nn
import scipy.sparse as sp
import itertools

data=torch.zero_(torch.Tensor([1,2,3,4,5,6,7,8,9]))
print(data)

edge_index=[[0,1],[0,2],[0,3],[1,0],[1,2],[2,0],[2,1],[2,3],[2,4],[3,0],[3,2],[4,2]]
edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
print(edge_index)

#adj = sp.coo_matrix((np.ones(len(edge_index)),(edge_index[:,0],edge_index[:,1])),shape=(5,5),dtype=torch.float32)

adjacency = sp.coo_matrix((np.ones(len(edge_index)),
                                   (edge_index[:, 0], edge_index[:, 1])),
                    shape=(5, 5), dtype="float32")

print(adjacency)