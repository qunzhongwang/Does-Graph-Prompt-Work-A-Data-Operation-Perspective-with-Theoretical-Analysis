import os
import time

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch.utils.data import TensorDataset
from prompt_graph_data import load4link_prediction_multi_graph

from gnn_models import GAT, GCN

DS = "ENZYMES"


class PreTrain(torch.nn.Module):
    def __init__(self, dataset, gnn_type='GCN', dim=25, gln=2, num_epoch=100, device=0):
        super().__init__()
        self.optimizer = None
        self.gnn = None
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset
        self.gnn_type = gnn_type
        self.num_layer = gln
        self.epochs = num_epoch
        self.dim = dim


    def initialize_gnn(self):
        if self.gnn_type == 'GAT':
            self.gnn = GAT(dim=self.input_dim, num_layer=self.num_layer, n_rank=0)
        elif self.gnn_type == 'GCN':
            self.gnn = GCN(dim=self.input_dim, num_layer=self.num_layer, n_rank=0)
        else:
            raise ValueError(f"Unsupported GNN type: {self.gnn_type}")
        print(self.gnn)
        self.gnn.to(self.device)
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=0.001, weight_decay=0.00005)


class Edgepred(PreTrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataloader = self.generate_loader_data()
        self.initialize_gnn()
        self.graph_pred_linear = torch.nn.Linear(self.dim, self.dim).to(self.device)

    def generate_loader_data(self):
        if self.dataset_name in ['PubMed', 'CiteSeer', 'Cora', 'Computers', 'Photo']:
            pass
            # # self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_single_graph(
            # #     self.dataset_name)
            # self.data.to(self.device)
            # edge_index = edge_index.transpose(0, 1)
            # data = TensorDataset(edge_label, edge_index)
            # return DataLoader(data, batch_size=64, shuffle=True)

        elif self.dataset_name in ['MUTAG', 'ENZYMES', 'COLLAB', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COX2',
                                   'BZR', 'PTC_MR']:
            self.data, edge_label, edge_index, self.input_dim, self.output_dim = load4link_prediction_multi_graph(
                self.dataset_name)
            self.data.to(self.device)
            edge_index = edge_index.transpose(0, 1)
            data = TensorDataset(edge_label, edge_index)
            return DataLoader(data, batch_size=64, shuffle=True)
        self.dim = self.dim
        self.initialize_gnn()
        self.graph_pred_linear = torch.nn.Linear(self.dim, self.dim).to(self.device)

    def pretrain_one_epoch(self):
        accum_loss, total_step = 0, 0
        device = self.device

        criterion = torch.nn.BCEWithLogitsLoss()

        self.gnn.train()
        for step, (batch_edge_label, batch_edge_index) in enumerate(self.dataloader):
            self.optimizer.zero_grad()

            batch_edge_label = batch_edge_label.to(device)
            batch_edge_index = batch_edge_index.to(device)

            out = self.gnn(self.data.x, self.data.edge_index,self.data.batch)
            node_emb = self.graph_pred_linear(out)

            batch_edge_index = batch_edge_index.transpose(0, 1)
            batch_pred_log = self.gnn.decode(node_emb, batch_edge_index).view(-1)
            loss = criterion(batch_pred_log, batch_edge_label)

            loss.backward()
            self.optimizer.step()

            accum_loss += float(loss.detach().cpu().item())
            total_step += 1

        return accum_loss / total_step

    def pretrain(self):
        num_epoch = self.epochs
        train_loss_min = 1000000
        for epoch in range(1, num_epoch + 1):
            st_time = time.time()
            train_loss = self.pretrain_one_epoch()
            print(f"[Pretrain] Epoch {epoch}/{num_epoch} | Train Loss {train_loss:.5f} | "
                  f"Cost Time {time.time() - st_time:.3}s")

            if train_loss_min > train_loss:
                train_loss_min = train_loss
                folder_path = f"./pre_trained_model/{self.dataset_name}"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                torch.save(self.gnn.state_dict(),
                           f"./pre_trained_model/{self.dataset_name}/linkpredict_{self.gnn_type}_{self.feature_dim}_{self.num_layer}.pth")

                print("+++model saved ! {}.{}.{}.{}.pth".format(self.dataset_name, 'link_predict', self.gnn_type,
                                                                str(self.hid_dim) + 'hidden_dim'))


def load_pre_train(gnn_type, feature_dim, num_layer, dataset_name):
    model_path = f"./pre_trained_model/{dataset_name}/linkpredict_{gnn_type}_{feature_dim}_{num_layer}.pth"

    if os.path.exists(model_path):
        model = Edgepred(dataset_name, gnn_type=feature_dim, dim=feature_dim, gln=num_layer)
        model.gnn.load_state_dict(torch.load(model_path))
        print(f"Loaded pre-trained model from {model_path}")
        return model.gnn
    else:
        print(f"Pre-trained model not found. Training a new model...")
        pretrained = Edgepred(dataset_name, gnn_type=gnn_type, dim=feature_dim, gln=feature_dim)
        pretrained.pretrain()
        input()
        # model.pretrain()
        # return model.gnn


def pre_train_model(train_type, model_type, feature_dim, num_layer, num_rank):
    if train_type not in ["N", "P"]:
        raise ValueError("Invalid type. Type must be 'N' or 'P'")
    if train_type == "N":
        if model_type == "GCN":
            model = GCN(dim=feature_dim, num_layer=num_layer, n_rank=num_rank)
            print("num of layer:", num_layer)
            print(model.tar_rank)
        else:
            model = GAT(dim=feature_dim, num_layer=num_layer, n_rank=num_rank)
    else:
        model = load_pre_train(model_type, feature_dim, num_layer, dataset_name=DS)
        model.re_rank(num_rank)
    return model


if __name__ == '__main__':
    pre_train_model("P","GCN",3,3,5)