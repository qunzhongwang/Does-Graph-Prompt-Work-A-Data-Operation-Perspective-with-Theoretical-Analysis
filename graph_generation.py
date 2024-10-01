import random

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset


def generate_random_graph(num_features, num_nodes_lo, num_nodes_hi, alpha):
    num_nodes = random.randint(num_nodes_lo, num_nodes_hi)
    x = torch.randn((num_nodes, num_features))
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < alpha:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    data = Data(x=x, edge_index=edge_index)
    return data


def load_from_dataset(dataset_name="MUTAG", batch_size=1):
    if dataset_name not in ["ENZYMES", "MUTAG", "NCI1","DD","NCI109"]:
        raise ValueError("Invalid type. Type must be ENZYMES, MUTAG, NCI1")
    dataset = TUDataset(root='dataset/' + dataset_name, name=dataset_name)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in loader:
        num_graphs = batch.num_graphs
        if num_graphs > 1:
            idx = random.randint(0, num_graphs - 1)
            subgraph = batch[idx]
        else:
            subgraph = batch
        graph = Data(x=subgraph.x, edge_index=subgraph.edge_index)
        yield graph


def graph_generate(type, num_features, num_nodes_lo, num_nodes_hi, alpha, dataset_name):
    if type not in ["S", "R", "B"]:
        raise ValueError("Invalid type. Type must be 'S' or 'R' or 'B'")
    if type == "S":
        return generate_random_graph(num_features, num_nodes_lo, num_nodes_hi, alpha)
    elif type == "R":
        dataset_gen = load_from_dataset(dataset_name)
        return next(dataset_gen)
    elif type == "B":
        if random.random() < 0.5:
            return generate_random_graph(num_features, num_nodes_lo, num_nodes_hi, alpha)
        else:
            dataset_gen = load_from_dataset()
            return next(dataset_gen)


if __name__ == "__main__":
    pass
