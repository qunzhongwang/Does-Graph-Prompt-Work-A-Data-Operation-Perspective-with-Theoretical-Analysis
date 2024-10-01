import torch
import pickle as pk
from random import shuffle
import random
from torch_geometric.datasets import Planetoid, Amazon, Reddit, WikiCS, Flickr
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
from torch_geometric.loader.cluster import ClusterData
from torch_geometric.data import Data,Batch
from torch_geometric.utils import negative_sampling
import os


def load4link_prediction_multi_graph(dataset_name, num_per_samples=1):
    if dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2', 'BZR',
                        'PTC_MR']:
        dataset = TUDataset(root='data/TUDataset', name=dataset_name)

    input_dim = dataset.num_features
    output_dim = 2  # link prediction的输出维度应该是2，0代表无边，1代表右边
    data = Batch.from_data_list(dataset)

    r"""Perform negative sampling to generate negative neighbor samples"""
    if data.is_directed():
        row, col = data.edge_index
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = data.edge_index

    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges * num_per_samples,
    )

    edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([torch.ones(data.num_edges), torch.zeros(neg_edge_index.size(1))], dim=0)

    return data, edge_label, edge_index, input_dim, output_dim
