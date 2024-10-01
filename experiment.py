import datetime

import pandas as pd
from tqdm import tqdm

import experiments_docker

FD = 0
NE = 0
NL = 0
RR = 0
ALPHA = 0


def experiment_1_1(model_type, prompt_type):
    loss_mean_list = []
    loss_var_list = []
    ratio_list = []

    num_4_loss = NE
    num_layer = NL
    feature_dim = FD
    trans_dense = ALPHA
    para_list = []

    for rank_to_reduce in tqdm(para_list, desc="Loop for Different PARA", colour='green'):
        loss_mean, loss_var = experiments_docker.experiment_1_docker(num_layer, rank_to_reduce,
                                                                     feature_dim, trans_dense,
                                                                     num_4_loss,
                                                                     model_type=model_type,
                                                                     prompt_type=prompt_type)
        loss_mean_list.append(loss_mean)
        loss_var_list.append(loss_var)
    print("process experiment_1_1" + model_type + prompt_type + "loaded")


def experiment_1_2(model_type, prompt_type):
    loss_mean_list = []
    loss_var_list = []
    ratio_list = []

    num_4_loss = NE
    num_layer = NL
    rank_to_reduce = RR
    trans_dense = ALPHA
    para_list = []

    for feature_dim in tqdm(para_list, desc="Loop for Different PARA", colour='green'):
        loss_mean, loss_var, ratio_mean, ratio_var = experiments_docker.experiment_1_docker(num_layer, rank_to_reduce,
                                                                                            feature_dim, trans_dense,
                                                                                            num_4_loss,
                                                                                            model_type=model_type,
                                                                                            prompt_type=prompt_type)
        loss_mean_list.append(loss_mean)
        loss_var_list.append(loss_var)
        ratio_list.append(ratio_mean)
    print("process experiment_1_2" + model_type + prompt_type + "loaded")


def experiment_1_3(model_type, prompt_type):
    loss_mean_list = []
    loss_var_list = []
    ratio_list = []

    num_4_loss = NE

    feature_dim = FD
    rank_to_reduce = RR
    trans_dense = ALPHA
    para_list = []
    print(para_list)

    for num_layer in tqdm(para_list, desc="Loop for Different PARA", colour='green'):
        loss_mean, loss_var = experiments_docker.experiment_1_docker(num_layer, rank_to_reduce,
                                                                     feature_dim, trans_dense,
                                                                     num_4_loss,
                                                                     model_type=model_type,
                                                                     prompt_type=prompt_type)

        loss_mean_list.append(loss_mean)
        loss_var_list.append(loss_var)

    print("process experiment_1_2" + model_type + prompt_type + "loaded")


def experiment_1_4(model_type, prompt_type):
    loss_mean_list = []
    loss_var_list = []
    ratio_list = []
    num_layer = 1
    feature_dim = 25
    rank_to_reduce = 10

    num_4_loss = 2

    para_list = [0.5 + _ / 20 for _ in range(5)]

    for trans_dense in tqdm(para_list, desc="Loop for Different PARA", colour='green'):
        loss_mean, loss_var, ratio_mean, ratio_var = experiments_docker.experiment_1_docker(num_layer, rank_to_reduce,
                                                                                            feature_dim, trans_dense,
                                                                                            num_4_loss,
                                                                                            model_type=model_type,
                                                                                            prompt_type=prompt_type)
        loss_mean_list.append(loss_mean)
        loss_var_list.append(loss_var)
        ratio_list.append(ratio_mean)

    data = list(zip(range(1, 1 + len(para_list)), loss_mean_list, loss_var_list, ratio_list))
    df = pd.DataFrame(data, columns=['Index', 'Loss', 'Loss Variance', 'Ratio'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv('result/experiment_1_4' + model_type + prompt_type + f'{timestamp}_trans_dense_loss.csv', index=False,
              encoding='utf-8-sig')
    print("process experiment_1_4 loaded")


def experiment_2_1(model_type, prompt_type):
    chi2_stat_list = []
    p_value_list = []
    NUM = 0

    num_layer = NL
    feature_dim = FD
    num_samples = NUM

    rank_to_reduce_list = []

    for rank_to_reduce in tqdm(rank_to_reduce_list, desc="Loop for Different PARA", colour='green'):
        chi2_stat, p_value = experiments_docker.experiment_2_docker(num_samples, rank_to_reduce,
                                                                    num_layer,
                                                                    feature_dim,
                                                                    model_type=model_type,
                                                                    prompt_type=prompt_type)

        chi2_stat_list.append(chi2_stat)
        p_value_list.append(p_value)
    print("process experiment_2_1 loaded")


def experiment_3_1(model_type, prompt_type):
    """
    gpf
    :param model_type:
    :param prompt_type:
    :return:
    """
    loss_mean_list = []
    loss_var_list = []
    ratio_list = []

    num_4_loss = 4
    num_layer = 1
    feature_dim = 37
    trans_dense = 0.6
    rank_to_reduce = 0
    para_prompt = 1
    need_ref = False

    li = [5 + 3 * _ for _ in range(3)]
    para_list = [1, 2, 3]
    for i in li:
        para_list.append(i)

    for num_graph in tqdm(para_list, desc="Loop for Different PARA", colour='green'):
        loss_mean, loss_var = experiments_docker.experiment_3_docker(num_graph, num_layer,
                                                                     rank_to_reduce,
                                                                     feature_dim, need_ref,
                                                                     trans_dense,
                                                                     num_4_loss,
                                                                     para_prompt,
                                                                     model_type=model_type,
                                                                     prompt_type=prompt_type)
        loss_mean_list.append(loss_mean)
        loss_var_list.append(loss_var)

    data = list(zip(range(4, 4 + len(para_list)), loss_mean_list, loss_var_list))
    df = pd.DataFrame(data, columns=['Rank to loss', 'Loss', 'Loss Variance'])

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    file_name = 'result/experi_3/1/' + model_type + '_' + prompt_type + f'_{timestamp}_num_graph.csv'
    df.to_csv(file_name, index=False,
              encoding='utf-8-sig')
    plot_single_file_loss(file_name, 'experi_3_1')

    print("process experiment_3_1" + model_type + prompt_type + " loaded")


def experiment_3_2(model_type, prompt_type):
    """
    allinone
    :param model_type:
    :param prompt_type:
    :return:
    """
    loss_mean_list = []
    loss_var_list = []
    ratio_list = []

    num_4_loss = 1
    rank_to_reduce = 0
    num_layer = 1
    feature_dim = 25
    trans_dense = 0.7
    need_ref = False

    num_graph_list = [30 + 40 * _ for _ in range(4)]
    para_prompt_list = [5 + 2 * _ for _ in range(4)]

    results = {}

    for num_graph in tqdm(num_graph_list, desc="Loop for Different PARA 1", colour='green'):
        if num_graph not in results:
            results[num_graph] = {}
        for para_prompt in tqdm(para_prompt_list, desc="Loop for Different PARA 2", colour='blue'):
            loss_mean, loss_var, ratio_mean, ratio_var = experiments_docker.experiment_3_docker(num_graph, num_layer,
                                                                                                rank_to_reduce,
                                                                                                feature_dim, need_ref,
                                                                                                trans_dense,
                                                                                                num_4_loss,
                                                                                                para_prompt,
                                                                                                model_type=model_type,
                                                                                                prompt_type=prompt_type)
            results[num_graph][para_prompt] = {'mean': loss_mean, 'var': loss_var}

    data_list = []
    for num_graph, prompt_dict in results.items():
        for prompt_para, metrics in prompt_dict.items():
            data_list.append({
                'num_graph': num_graph,
                'prompt_para': prompt_para,
                'mean': metrics['mean'],
                'var': metrics['var']
            })

    df = pd.DataFrame(data_list)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = 'result/experi_3/2/' + model_type + '_' + prompt_type + f'_{timestamp}_numg_prop_loss.csv'
    df.to_csv(file_name, index=False,
              encoding='utf-8-sig')
    plot_3d_mean_with_var(file_name, title='3D Plot of Mean with Variance')

    print("process experiment_3_2" + model_type + prompt_type + " loaded")


if __name__ == '__main__':
    # plot_3d_mean_with_var('result/experi_3/2/GCN_gpf-plus_20240924_100031_numg_prop_loss.csv',
    # title='3D Plot of Mean with Variance')
    pass
