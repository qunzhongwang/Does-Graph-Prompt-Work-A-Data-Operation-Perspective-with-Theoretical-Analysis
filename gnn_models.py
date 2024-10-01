import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, dim, num_layer, n_rank):
        super().__init__()
        GraphConv = GCNConv
        input_dim = dim
        hid_dim = dim
        out_dim = dim
        if num_layer == 1:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, out_dim, bias=False)])
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList(
                [GraphConv(input_dim, hid_dim, bias=False), GraphConv(hid_dim, out_dim, bias=False)])
        else:
            layers = [GraphConv(input_dim, hid_dim, bias=False)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim, bias=False))
            layers.append(GraphConv(hid_dim, out_dim, bias=False))
            self.conv_layers = torch.nn.ModuleList(layers)
        self.dim = dim
        self.pool = global_add_pool
        self.rank = []
        self.calcu_rank()
        self.n_rank = n_rank
        rank = [input_dim for i in range(num_layer)]
        rank[-1] = rank[-1] - n_rank
        self.tar_rank = rank
        self.re_rank(None)

    def forward(self, x, edge_index, batch):

        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(x, batch=batch)
        return graph_emb

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    # TODO

    def branch(self, x, edge_index):

        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
            if idx == 1 or idx == 2:
                x = F.leaky_relu(x)
        x = self.conv_layers[-1](x, edge_index)

        return x

    def calcu_rank(self):
        self.rank = []
        for conv in self.conv_layers:
            weight = conv.lin.weight.detach().numpy()
            rank = np.linalg.matrix_rank(weight)
            self.rank.append(rank)

    def re_rank(self, num_rank=None):
        if num_rank is None:
            pass
        else:
            self.tar_rank = [x - num_rank for x in self.tar_rank]

        for i, conv in enumerate(self.conv_layers):
            if i == 0:
                weight = conv.lin.weight.detach().numpy()
                U, S, Vt = np.linalg.svd(weight, full_matrices=False)
                U_r = U[:, :self.tar_rank[i]]
                S_r = np.diag(S[:self.tar_rank[i]])
                Vt_r = Vt[:self.tar_rank[i], :]
                reduced_weight = U_r @ S_r @ Vt_r
                conv.lin.weight = torch.nn.Parameter(torch.tensor(reduced_weight, dtype=torch.float32))
        self.rank = self.tar_rank


class GAT(torch.nn.Module):
    def __init__(self, dim, num_layer, n_rank):
        super().__init__()
        GraphConv = GATConv
        input_dim = dim
        hid_dim = dim
        out_dim = dim
        if num_layer == 1:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, out_dim)])
        elif num_layer == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim), GraphConv(hid_dim, out_dim)])
        else:
            layers = [GraphConv(input_dim, hid_dim, bias=False)]
            for i in range(num_layer - 2):
                layers.append(GraphConv(hid_dim, hid_dim, bias=False))
            layers.append(GraphConv(hid_dim, out_dim, bias=False))
            self.conv_layers = torch.nn.ModuleList(layers)
        self.dim = dim
        self.pool = global_add_pool
        self.rank = []
        self.calcu_rank()

        self.n_rank = n_rank
        self.tar_rank = [input_dim - n_rank for i in range(num_layer)]
        self.re_rank(None)

    def forward(self, x, edge_index, batch):

        for idx, conv in enumerate(self.conv_layers[0:-1]):
            x = conv(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(x, batch=batch)
        return graph_emb

    def calcu_rank(self):
        self.rank = []
        for conv in self.conv_layers:
            weight_src = conv.lin.weight.detach().numpy()
            rank = np.linalg.matrix_rank(weight_src)
            self.rank.append(rank)

    def re_rank(self, num_rank=None):
        if num_rank is None:
            pass
        else:
            self.tar_rank = [x - num_rank for x in self.tar_rank]
            for i, conv in enumerate(self.conv_layers):
                weight_src = conv.lin.weight.detach().numpy()
                U, S, Vt = np.linalg.svd(weight_src, full_matrices=False)
                U_r = U[:, :self.tar_rank[i]]
                S_r = np.diag(S[:self.tar_rank[i]])
                Vt_r = Vt[:self.tar_rank[i], :]
                reduced_weight = U_r @ S_r @ Vt_r
                conv.lin.weight = torch.nn.Parameter(torch.tensor(reduced_weight, dtype=torch.float32))
        self.rank = self.tar_rank


if __name__ == '__main__':
    model = GAT(5, 1, 2)

    # pass
