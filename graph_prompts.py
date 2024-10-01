import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import utills

class GPF(torch.nn.Module):
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = torch.nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, graph_batch: Batch):
        num_nodes = graph_batch.num_nodes
        global_emb_expanded = self.global_emb.expand(num_nodes, -1)
        graph_batch.x = graph_batch.x + global_emb_expanded
        return graph_batch

class GPF_plus(torch.nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPF_plus, self).__init__()
        self.p_list = torch.nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = torch.nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, graph_batch: Batch):
        score = self.a(graph_batch.x)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)
        graph_batch.x = graph_batch.x + p
        return graph_batch

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None):
        super(LightPrompt, self).__init__()
        self.token_num = token_num_per_group
        self.inner_prune = inner_prune
        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])
        self.link_vec = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])
        self.token_init(init_method="kaiming_uniform")

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            for vec in self.link_vec:
                torch.nn.init.kaiming_uniform_(vec, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self):
        pg_list = []
        for i, (vec, tokens) in enumerate(zip(self.link_vec, self.token_list)):

            token_dot = torch.mm(vec, torch.transpose(vec, 0, 1))
            token_sim = torch.sigmoid(token_dot)  # 0-1
            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()
            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))
        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch

class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, cross_prune, inner_prune):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune)
        self.cross_prune = cross_prune

    def add(self, graph_batch: Batch):
        pg = self.inner_structure_update()

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]
        re_graph_list = []

        for g, vec in zip(Batch.to_data_list(graph_batch), self.link_vec):
            g_edge_index = g.edge_index + token_num
            cross_dot = torch.mm(vec, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)

            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch

class Allinone_plus(LightPrompt):
    def __init__(self, token_dim, token_num, cross_mat_num, cross_prune, inner_prune):
        super(Allinone_plus, self).__init__(token_dim, token_num, 1, inner_prune)  # only has one prompt graph.
        self.cross_prune = cross_prune
        self.mat_list = nn.ParameterList(
            [nn.Parameter(torch.Tensor(token_num, token_dim)) for i in range(cross_mat_num)])
        self.task_head = nn.Linear(cross_mat_num, 1)
        self.mat_init(init_method="kaiming_uniform")

    def mat_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":

            for mat in self.mat_list:
                torch.nn.init.kaiming_uniform_(mat, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            torch.nn.init.kaiming_uniform_(self.task_head.weight, nonlinearity='leaky_relu', mode='fan_in', a=0.01)

        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def get_link_dot(self, x):
        cross_dot_list = []
        num_graph = x.size(0)
        for mat in self.mat_list:
            cross_dot_list.append(torch.mm(mat, torch.transpose(x, 0, 1)))
        cross_dot = torch.zeros(self.token_num, num_graph)

        for i in range(self.token_num):
            for j in range(num_graph):
                vec = torch.tensor([mat[i][j] for mat in cross_dot_list])
                value = self.task_head(vec)
                cross_dot[i][j] = value

        return cross_dot

    def add(self, graph_batch: Batch):
        pg = self.inner_structure_update()

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]
        re_graph_list = []

        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num
            cross_dot = self.get_link_dot(g.x)
            cross_sim = torch.sigmoid(cross_dot)

            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num

            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)
        return Batch.from_data_list(re_graph_list)


if __name__ == '__main__':
    pass
