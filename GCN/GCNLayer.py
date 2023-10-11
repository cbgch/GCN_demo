# from torch import nn
# import torch
#
# class GCNLayer(nn.Module):
#     def __init__(self,inputFeat,outputFeat,bias=True):
#         super(GCNLayer,self).__init__()
#         #定义权重矩阵
#         # nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面
#         self.weight=nn.Parameter(torch.Tensor(inputFeat,outputFeat))
#         if bias:
#             self.bias=nn.Parameter(torch.zero_(outputFeat))
#         else:
#             self.bias=None
#         self.reset_parameter()
#
#     #???
#     def reset_parameter(self):
#         nn.init.xavier_uniform_(self.weight)
#
#     def forward(self,g,h,fn):
#         with g.local_scope():
#             h=torch.matmul(h,self.weight)
#             g.ndata['h']=g.ndata['norm'] * h
#             g.update_all(message_func=fn.copy_u('h','m'),reduce_func=fn.sum('m','h'))
#             h=g.ndata['h']


#*****************************************************************************************************************
import itertools
import os
import os.path as osp
import pickle
import urllib
from collections import namedtuple
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
#%matplotlib inline

Data=namedtuple('Data',['x','y','adjacency','train_mask','val_mask','trian_mask'])

def tensor_from_numpy(x,device):
    return torch.from_numpy(x).to(device)

class CoraData(object):
    filenames = ["ind.cora.{}".format(name) for name in
                 ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']]

    def __init__(self,root='./data',rebuild=False):
        self.dataroot=root
        save_file=osp.join(root,'ch5_cached.pkl')
        if osp.exists(save_file) and not rebuild:
            print("Using Cached file: {}".format(save_file))
            self._data=pickle.load(open(save_file,'rb'))    #从pickle格式的文件中读取数据并转换为Python的类型
        else:
            self._data=self.process_data()
            with open(save_file,'rb') as f:
                pickle.dump(self._data,f)
            print("Cached file: {}".format(save_file))
    @property   #property装饰器使得调用该函数时可以直接obj.data,不需要加括号
    def data(self):
        return self._data

    def process_data(self):
        print("Process data ...")
        _, tx, allx, y, ty, ally, graph, test_index = [self.read_data(
            osp.join(self.data_root, name)) for name in self.filenames]
        train_index=np.arrange(y.shape[0])
        val_index=np.arange(y.shape[0],y.shape[0]+500)
        sorted_test_index = sorted(test_index)

        x = np.concatenate((allx, tx), axis=0)  # 节点特征
        y = np.concatenate((ally, ty), axis=0).argmax(axis=1)  # 标签

        x[test_index] = x[sorted_test_index]   #重新排序
        y[test_index] = y[sorted_test_index]
        num_nodes = x.shape[0]

        train_mask = np.zeros(num_nodes, dtype=np.bool)  # 训练集
        val_mask = np.zeros(num_nodes, dtype=np.bool)  # 验证集
        test_mask = np.zeros(num_nodes, dtype=np.bool)  # 测试集
        train_mask[train_index] = True
        val_mask[val_index] = True
        test_mask[test_index] = True

        adjacency = self.build_adjacency(graph)
        print("Node's feature shape: ", x.shape)
        print("Node's label shape: ", y.shape)
        print("Adjacency's shape: ", adjacency.shape)
        print("Number of training nodes: ", train_mask.sum())
        print("Number of validation nodes: ", val_mask.sum())
        print("Number of test nodes: ", test_mask.sum())

        return Data(x=x, y=y, adjacency=adjacency,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    @staticmethod
    def build_adjacency(adj_dict):
        edge_index=[]
        number_nodes=len(adj_dict)
        for src,dst in adj_dict.items():
            edge_index.extend([src, v] for v in dst)
            edge_index.extend([v, src] for v in dst)
        #itertools.groupby()将可迭代对象中相邻元素中相同的挑出来  aaabbbcccccaa --> abca
        edge_index = list(k for k, _ in itertools.groupby(sorted(edge_index)))
        edge_index = np.asarray(edge_index)
        adjacency = sp.coo_matrix((np.ones(len(edge_index))),(edge_index[:,0],edge_index[:,1]),
                                  shape=(number_nodes,number_nodes),dtype=torch.float32)
        return adjacency




