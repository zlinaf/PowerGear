import imp
from typing import Union, Tuple
import numpy as np
import pandas as pd
import __init__
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, Linear, BatchNorm1d,LSTM 
from torch_sparse import SparseTensor, matmul, masked_select_nnz
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size, OptPairTensor
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
import __init__
from utils.base_func import masked_edge_index,masked_edge_attr

class HECConv(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,dim = 1 , num_relation = 1, 
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(HECConv, self).__init__(aggr=aggr, **kwargs)
        self.num_realation = num_relation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relation_weight = nn.ModuleList()
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        for i in range(num_relation):
            self.relation_weight.append(Linear(in_channels[0],out_channels))
        #self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.attr_fc = Linear(dim,in_channels[1],bias = False)
        self.reset_parameters()
        self.dim = dim
    def reset_parameters(self):
        for fc in self.relation_weight:
            fc.reset_parameters()
        self.lin_r.reset_parameters()
        self.attr_fc.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, edge_type: OptTensor = None,size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = torch.zeros(x[0].size(0), self.out_channels, device=x[0].device)
        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        if self.dim >1 :
            edge_weight = self.attr_fc(edge_weight)
        for i, conv in enumerate(self.relation_weight):
            tmp = masked_edge_index(edge_index, edge_type == i)
            tmp_out = self.propagate(tmp, x=x, edge_weight=masked_edge_attr(edge_weight, edge_type == i),
                                size=size)
            tmp_out = conv(tmp_out)
            out = out + tmp_out
        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        print("=======")
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
        
        
class HECConv_no_relation(MessagePassing):
    def __init__(self, in_channels: Union[int, Tuple[int,int]], out_channels: int,dim = 1 , num_relation = 1, 
                 aggr: str = 'add', bias: bool = True, **kwargs):
        super(HECConv_no_relation, self).__init__(aggr=aggr, **kwargs)
        self.num_realation = num_relation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relation_weight = nn.ModuleList()
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        #self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)
        self.attr_fc = Linear(dim,out_channels,bias = False)
        self.reset_parameters()
        self.dim = dim
    def reset_parameters(self):
        self.lin_r.reset_parameters()
        self.attr_fc.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_weight: OptTensor = None, edge_type: OptTensor = None,size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = torch.zeros(x[0].size(0), self.out_channels, device=x[0].device)
        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        if self.dim >1 :
            edge_weight = self.attr_fc(edge_weight)
        tmp_out = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight,
                            size=size)
        out = out+tmp_out
        x_r = x[1]
        if x_r is not None:
            out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        print("=======")
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)