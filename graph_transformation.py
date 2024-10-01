import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

import graph_generation
import utills

Dense = 0


def node_reduce(data, beta):
    node_mask = torch.bernoulli(torch.full((data.num_nodes,), 1 - 0.8 * beta)).bool()
    transformed_data = data.subgraph(node_mask)
    return transformed_data


def node_addition(data, beta):
    num_nodes = data.num_nodes
    new_nodes = int(beta * data.num_nodes)
    new_node_features = torch.randn((new_nodes, data.x.size(1)))
    augmented_features = torch.cat((data.x, new_node_features), dim=0)

    new_edge_index = []
    for i in range(num_nodes, num_nodes + new_nodes):
        for j in range(i):
            if np.random.uniform(0, 1) < beta * 0.95:
                new_edge_index.append([i, j])
                new_edge_index.append([j, i])

    new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
    augmented_edge_index = torch.cat((data.edge_index, new_edge_index), dim=1)
    augmented_data = Data(x=augmented_features, edge_index=augmented_edge_index)
    return augmented_data


def feature_noise(data, beta):
    var = 0.2
    noise = torch.randn_like(data.x) * beta * var
    data.x += noise
    return data


def link_delete(data, beta):
    edge_mask = torch.bernoulli(torch.full((data.num_edges,), 1 - 0.2 * beta)).bool()
    data.edge_index = data.edge_index[:, edge_mask]
    data.edge_attr = data.edge_attr[edge_mask] if data.edge_attr is not None else None
    return data


def link_addition(data, beta):
    proba = beta * 0.25
    num_nodes = data.num_nodes
    new_edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if np.random.uniform(0, 1) < proba:
                new_edge_index.append([i, j])
    if new_edge_index:
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
        data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    return data


def graph_transform(data, beta):
    """
    random
    """
    transformed_data = data.clone()
    transformed_data = node_addition(transformed_data, beta)
    transformed_data = node_reduce(transformed_data, beta)
    transformed_data = feature_noise(transformed_data, beta)
    transformed_data = link_delete(transformed_data, beta)
    transformed_data = link_addition(transformed_data, beta)
    return transformed_data


def edge_node_transform(data):
    edge_index = data.edge_index
    node_features = data.x
    num_edges = edge_index.size(1)
    new_node_features = []
    edge_map = {}
    for i in range(num_edges):
        src, dst = edge_index[0, i], edge_index[1, i]
        if (src.item(), dst.item()) not in edge_map and (dst.item(), src.item()) not in edge_map:
            new_feature = node_features[src] + node_features[dst]
            new_node_features.append(new_feature)
            edge_map[(src.item(), dst.item())] = len(new_node_features) - 1

    new_node_features = torch.stack(new_node_features, dim=0)
    new_edge_index = []

    for (src1, dst1), idx1 in edge_map.items():
        for (src2, dst2), idx2 in edge_map.items():
            if idx1 < idx2:
                if src1 == src2 or src1 == dst2 or dst1 == src2 or dst1 == dst2:
                    new_edge_index.append([idx1, idx2])
    new_edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
    new_data = Data(x=new_node_features, edge_index=new_edge_index)
    return new_data


def graph_det_transform(data):
    edge_index = data.edge_index
    node_features = data.x
    num_nodes = node_features.size(0)
    node_degrees = degree(edge_index[0], num_nodes=num_nodes)
    high_degree_threshold = node_degrees.mean() + 0.75 * node_degrees.std()
    edges_to_keep = []

    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i], edge_index[1, i]
        if node_degrees[src] <= high_degree_threshold and node_degrees[dst] <= high_degree_threshold:
            edges_to_keep.append(i)
        elif np.random.uniform(0, 1) > 0.7:  # 随机保留一些边
            edges_to_keep.append(i)

    new_edge_index = edge_index[:, edges_to_keep]
    mean_feature = node_features.mean(dim=0)
    feature_std = node_features.std(dim=0)
    deviation_threshold = mean_feature + feature_std

    for i in range(node_features.size(0)):
        if (node_features[i] > deviation_threshold).any():
            node_features[i] = mean_feature + (node_features[i] - mean_feature) * 0.75

    low_degree_threshold = node_degrees.mean() - 0.75 * node_degrees.std()
    low_degree_nodes = (node_degrees <= low_degree_threshold).nonzero(as_tuple=True)[0]

    added_edges = []
    for i in range(len(low_degree_nodes)):
        for j in range(i + 1, len(low_degree_nodes)):
            if np.random.uniform(0, 1) > 0.7:  # 随机决定是否添加边
                added_edges.append([low_degree_nodes[i].item(), low_degree_nodes[j].item()])
    if added_edges:
        added_edges = torch.tensor(added_edges, dtype=torch.long).t().contiguous()
        new_edge_index = torch.cat([new_edge_index, added_edges], dim=1)

    new_data = Data(x=node_features, edge_index=new_edge_index)
    return new_data


def sub_graph_recons(data):
    pass


def find_best_ever_embe(embe_type, graph_list, model, dense=Dense):
    if embe_type == "N":
        graph_trans = [graph_transform(graph, beta=dense) for graph in
                       graph_list]
        best_ever_embe = graph_trans
    else:
        pass
    return best_ever_embe


if __name__ == '__main__':
    graph = graph_generation.generate_random_graph(5, 8, 15, 0.3)
    utills.plot_graph(graph)
    new_graph = graph_det_transform(graph)
    utills.plot_graph(new_graph)
