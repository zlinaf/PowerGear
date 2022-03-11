import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data
from torch._six import container_abcs, string_classes, int_classes

import torch
from torch import Tensor
from torch_sparse import SparseTensor, cat
import torch_geometric
from torch_geometric.data import Data




class Batch(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (disconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """
    def __init__(self, batch=None, **kwargs):
        super(Batch, self).__init__(**kwargs)

        for key, item in kwargs.items():
            if key == 'num_nodes':
                self.__num_nodes__ = item
            elif key == 'num_edges':
                self.__num_edges__ = item
            else:
                self[key] = item

        self.node_batch = batch
        self.edge_batch = batch
        self.__data_class__ = Data
        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__num_nodes_list__ = None
        self.__num_graphs__ = None

    @staticmethod
    def from_data_list(data_list, follow_batch=[],edge_batch = False):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""
        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch()
        for key in data_list[0].__dict__.keys():
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None
        # if edge_batch:
        #     batch.__num_nodes__ = batch.__num_edges__

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['node_batch','edge_batch']:
            batch[key] = []

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        num_edges_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Treat 0-dimensional tensors as 1-dimensional.
                if isinstance(item, Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)

                batch[key].append(item)

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                cat_dims[key] = cat_dim
                if isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                    device = item.device()

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size, ), i, dtype=torch.long,
                                       device=device))
            
            if hasattr(data, '__num_edges__'):
                num_edges_list.append(data.__num_edges__)
            else:
                num_edges_list.append(None)
            num_edges = data.num_edges
        
            if hasattr(data, '__num_nodes__'):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)
            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long,
                                  device=device)
                batch.node_batch.append(item)
            if num_edges is not None:
                item = torch.full((num_edges, ), i, dtype=torch.long,
                                  device=device)
                batch.edge_batch.append(item)
        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.node_batch = None if len(batch.node_batch) == 0 else batch.node_batch
        batch.edge_batch = None if len(batch.edge_batch) == 0 else batch.edge_batch
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using `Batch.from_data_list()`.'))

        data_list = []
        for i in range(len(list(self.__slices__.values())[0]) - 1):
            data = self.__data_class__()

            for key in self.__slices__.keys():
                item = self[key]
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][i]
                    end = self.__slices__[key][i + 1]
                    item = item.narrow(dim, start, end - start)
                elif isinstance(item, SparseTensor):
                    for j, dim in enumerate(self.__cat_dims__[key]):
                        start = self.__slices__[key][i][j].item()
                        end = self.__slices__[key][i + 1][j].item()
                        item = item.narrow(dim, start, end - start)
                else:
                    item = item[self.__slices__[key][i]:self.
                                __slices__[key][i + 1]]
                    item = item[0] if len(item) == 1 else item

                # Decrease its value by `cumsum` value:
                cum = self.__cumsum__[key][i]
                if isinstance(item, Tensor):
                    if not isinstance(cum, int) or cum != 0:
                        item = item - cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value - cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item - cum

                data[key] = item

            if self.__num_nodes_list__[i] is not None:
                data.num_nodes = self.__num_nodes_list__[i]

            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        if self.__num_graphs__ is not None:
            return self.__num_graphs__
        return self.node_batch[-1].item() + 1


######DataLoader##########

class Collater(object):
    def __init__(self, follow_batch,edge_batch = False):
        self.follow_batch = follow_batch
        self.edge_batch = edge_batch

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch,edge_batch=self.edge_batch)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],edge_batch = False,
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch,edge_batch), **kwargs)


class DataListLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a python list.

    .. note::

        This data loader should be used for multi-gpu support via
        :class:`torch_geometric.nn.DataParallel`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DataListLoader, self).__init__(
            dataset, batch_size, shuffle,
            collate_fn=lambda data_list: data_list, **kwargs)


class DenseCollater(object):
    def collate(self, data_list):
        batch = Batch()
        for key in data_list[0].keys:
            batch[key] = default_collate([d[key] for d in data_list])
        return batch

    def __call__(self, batch):
        return self.collate(batch)


class DenseDataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    .. note::

        To make use of this data loader, all graphs in the dataset needs to
        have the same shape for each its attributes.
        Therefore, this data loader should only be used when working with
        *dense* adjacency matrices.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch (default: :obj:`False`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super(DenseDataLoader, self).__init__(
            dataset, batch_size, shuffle, collate_fn=DenseCollater(), **kwargs)
