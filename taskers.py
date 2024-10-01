import csv
import random

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader, Batch
from tqdm import tqdm

import gnn_models
import graph_generation
import graph_matching
import graph_prompts
import graph_transformation
import model_pretraining

LR = 0.0006
WD = 0.000005
SC = 4
TP = "N"
TE = "N"
ME = 1500
MD = 0.001
DS = ""
DM = 100


def save_to_csv(epochs, mean_loss, std_loss, filename, sample_interval=999, max_samples=20):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Epoch', 'Mean Loss', 'Std Loss'])
        total_epochs = len(epochs)
        sample_indices = np.arange(0, total_epochs, sample_interval)

        sample_indices = sample_indices[:max_samples]

        for i in sample_indices:
            row = [epochs[i], mean_loss[i], std_loss[i]]
            writer.writerow(row)


def prompt_train(p, model, graph_trans, graph_list, max_epochs=2500, min_delta=MD, patience=SC):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, p.parameters()), lr=0.0005, weight_decay=WD)
    avg_loss = None
    stopping_counter = 0

    best_loss = float('inf')
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        combined_graphs = list(zip(graph_trans, graph_list))
        loader = DataLoader(combined_graphs, batch_size=1)
        loss = torch.tensor(0.0, requires_grad=True)

        for batch_data in loader:
            trans_data, list_data = batch_data
            prompt_data = p.add(Batch.from_data_list([list_data]))
            trans_output = model(trans_data.x, trans_data.edge_index, trans_data.batch)
            prompt_output = model(prompt_data.x, prompt_data.edge_index, prompt_data.batch)
            loss = loss + graph_matching.graph_matching(trans_output, prompt_output)

        avg_loss = loss.item() / len(graph_trans)
        loss.backward()
        optimizer.step()

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            stopping_counter = 0
        else:
            stopping_counter += 1
        if stopping_counter >= patience:
            print(f"Epoch: {epoch}, Loss: {avg_loss}")
            print("Early stopping")
            break
        if (epoch + 1) % 200 == 0:
            print(f"Epoch: {epoch}, Loss: {avg_loss}")

    return avg_loss



def prompt_train_his(p, model, graph_trans, graph_list, max_epochs=2500, min_delta=0.0002, patience=4):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, p.parameters()), lr=0.0001, weight_decay=10 ** (-5))
    avg_loss = None
    stopping_counter = 0
    best_loss = float('inf')
    loss_history = []

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        combined_graphs = list(zip(graph_trans, graph_list))
        loader = DataLoader(combined_graphs, batch_size=1)
        loss = torch.tensor(0.0, requires_grad=True)

        for batch_data in loader:
            trans_data, list_data = batch_data
            prompt_data = p.add(Batch.from_data_list([list_data]))
            trans_output = model(trans_data.x, trans_data.edge_index, trans_data.batch)
            prompt_output = model(prompt_data.x, prompt_data.edge_index, prompt_data.batch)
            loss = loss + graph_matching.graph_matching(trans_output, prompt_output)

        avg_loss = loss.item() / len(graph_trans)
        loss_history.append(avg_loss)
        loss.backward()
        optimizer.step()

        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss
            stopping_counter = 0
        else:
            stopping_counter += 1
        if stopping_counter >= patience:
            print(f"Epoch: {epoch}, Loss: {avg_loss}")
            print("Early stopping")
            break
        if (epoch + 1) % 200 == 0:
            print(f"Epoch: {epoch}, Loss: {avg_loss}")

    return loss_history, avg_loss



def meta_experiment(num_graph, feature_dim, graph_size_lo, graph_size_hi, model_type, num_layer, num_rank, prompt_type
                    , max_epochs, epoch, hyper_para):
    def create_prompt():
        if prompt_type == "gpf":
            p = graph_prompts.GPF(feature_dim)
        elif prompt_type == "gpf-plus":
            p = graph_prompts.GPF_plus(feature_dim, hyper_para["prompt_num"])
        elif prompt_type == "All-in-one":
            p = graph_prompts.HeavyPrompt(token_dim=feature_dim, token_num=hyper_para["token_num"],
                                          inner_prune=hyper_para["inner_prune"],
                                          cross_prune=hyper_para["cross_prune"], )
        else:
            p = graph_prompts.Allinone_plus(token_dim=feature_dim, token_num=hyper_para["token_num"],
                                            cross_mat_num=hyper_para["cross_mat_num"],
                                            inner_prune=hyper_para["inner_prune"],
                                            cross_prune=hyper_para["cross_prune"])
        return p

    num_data_set = DM
    graph_list = [
        graph_generation.graph_generate(type="S", num_features=feature_dim, num_nodes_lo=graph_size_lo,
                                        num_nodes_hi=graph_size_hi,
                                        alpha=hyper_para["link_dense"], dataset_name=DS) for _ in
        range(num_data_set)]
    model = model_pretraining.pre_train_model(train_type=TP, feature_dim=feature_dim, num_layer=num_layer,
                                              num_rank=num_rank, model_type=model_type)

    loss_list = []
    for _ in tqdm(range(epoch), desc="FIND UPPER Loop", colour='red', leave=False):
        selected_graphs = random.sample(graph_list, num_graph)
        best_ever_embe = graph_transformation.find_best_ever_embe(TE, selected_graphs, model=model,
                                                                  dense=hyper_para["trans_dense"],
                                                                  )
        p = create_prompt()
        loss = prompt_train(p, model, best_ever_embe, graph_list, max_epochs=max_epochs)
        loss_list.append(loss)
        print(loss)
    print(loss_list)
    return loss_list


def container(aug_list, epoch=1):
    loss_list = meta_experiment(num_graph=aug_list["num_graph"],
                                feature_dim=aug_list["feature_dim"],
                                graph_size_lo=aug_list["graph_size_lo"],
                                graph_size_hi=aug_list["graph_size_hi"],
                                num_layer=aug_list["num_layer"], num_rank=aug_list["num_rank"],
                                max_epochs=aug_list["max_epochs"],
                                model_type=aug_list["model_type"],
                                prompt_type=aug_list["prompt_type"],
                                hyper_para=aug_list["hyper_para"], epoch=epoch)

    return loss_list


def multi_experiment_4_lower_bound(aug_list):
    loss_list = []
    loss_list = container(aug_list, epoch=aug_list["num_to_loop"])
    print(f"min-losses:{min(loss_list)}")
    return min(loss_list)


def multi_experiment_4_upper_bound(aug_list):
    loss_list = []
    loss_list = container(aug_list, epoch=aug_list["num_to_loop"])
    print(f"max-losses:{max(loss_list)}")
    return max(loss_list)


def multi_experiment_4_quantile_bound(aug_list):
    loss_list = container(aug_list, epoch=aug_list["num_to_loop"])
    loss_mean = np.mean(loss_list)
    print(f"quantile-losses:{loss_mean}")
    return min(loss_list)

def his_experiment(num_experiments=10, **train_params):
    all_loss_histories = []
    final_losses = []
    for i in range(num_experiments):
        if train_params['model_type'] == "GCN":
            model = gnn_models.GCN(dim=train_params['feature_dim'], num_layer=1, n_rank=0)
        else:
            model = gnn_models.GAT(dim=train_params['feature_dim'], num_layer=1, n_rank=0)

        graph_list = [graph_generation.graph_generate("R", num_features=train_params['feature_dim'], num_nodes_lo=8,
                                                      num_nodes_hi=12,
                                                      alpha=0.1, dataset_name="") for _ in range(1)]

        graph_trans = [graph_transformation.graph_transform(graph, beta=0.15) for graph in
                       graph_list]

        if train_params['prompt_type'] == "gpf":
            p = graph_prompts.GPF(train_params['feature_dim'])
        elif train_params['prompt_type'] == "gpf-plus":
            p = graph_prompts.GPF_plus(train_params['feature_dim'], 3)
        elif train_params['prompt_type'] == "All-in-one":
            p = graph_prompts.HeavyPrompt(token_dim=train_params['feature_dim'], token_num=3,
                                          inner_prune=0.3, cross_prune=0.7)
        else:
            p = graph_prompts.Allinone_plus(token_dim=train_params['feature_dim'], token_num=3,
                                            cross_mat_num=3,
                                            inner_prune=0.3, cross_prune=0.7)

        print(f"Running experiment {i + 1}/{num_experiments}")
        loss_history, final_loss = prompt_train_his(p=p, graph_list=graph_list, graph_trans=graph_trans, model=model,
                                                    max_epochs=20000, min_delta=0, patience=20000)

        all_loss_histories.append(loss_history)
        final_losses.append(final_loss)

    return None


def zero_error_construct():
    prompt_type = 'All-in-one'
    model_type = 'GCN'
    feature_dim = 0
    train_params = {}
    his_experiment(num_experiments=0, **train_params)


if __name__ == "__main__":
    pass