import datetime
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.subplots import make_subplots
from tqdm import tqdm

import taskers


def save_results_to_csv(results):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result/distribution_analysis_{timestamp}.csv"
    data_to_save = {
        "loss": [results["mean_loss"]],
        "variance": [results["variance"]],
    }
    df = pd.DataFrame(data_to_save)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


def experiment_1_docker(num_layer, rank_to_reduce, feature_dim, trans_dense, num_4_loss, model_type, prompt_type):
    losses = []

    aug_list = {
        "num_graph": 1,
        "feature_dim": feature_dim,
        "num_layer": num_layer,
        "num_rank": rank_to_reduce,
        "model_type": model_type,
        "prompt_type": prompt_type,
        "graph_size_lo": 40,
        "graph_size_hi": 50,
        "num_to_loop": 6,
        "need_expect_value": True,
        "max_epochs": 2000,
        "experiment_type": "single_graph",
        "hyper_para": {
            "link_dense": 0.1,
            "trans_dense": trans_dense,
            "prompt_num": 5,
            "token_num": 10,
            "inner_prune": 0.3,
            "cross_prune": 0.2,
            "cross_mat_num": 5,}
    }


    for _ in tqdm(range(num_4_loss), desc="Loop for different Upper LOSS", colour='blue'):
        loss = taskers.multi_experiment_4_upper_bound(aug_list)
        losses.append(loss)
    losses = np.array(losses)
    (loss_mean, loss_var) = (losses.mean(), losses.var())

    print(
        f"loss_mean:{loss_mean}, loss_var:{loss_var}\n with para:[rank_to_reduce:{rank_to_reduce}]")
    return loss_mean, loss_var


def experiment_2_docker(num_samples, rank_to_reduce, num_layer, feature_dim, model_type, prompt_type):
    def calculate_distributions(data_list, num_bins):
        pdf_mean = np.zeros(num_bins)
        pdf_std = np.zeros(num_bins)
        cdf_mean = np.zeros(num_bins)
        cdf_std = np.zeros(num_bins)
        all_data = np.concatenate(data_list)
        bins = np.linspace(min(all_data), max(all_data), num_bins + 1)

        pdfs = []
        cdfs = []

        for data in data_list:
            hist, _ = np.histogram(data, bins=bins, density=True)
            pdfs.append(hist)

            ecdf = np.cumsum(hist) / sum(hist)
            cdfs.append(ecdf)

        pdfs = np.array(pdfs)
        cdfs = np.array(cdfs)

        pdf_mean = np.mean(pdfs, axis=0)
        pdf_std = np.std(pdfs, axis=0)
        cdf_mean = np.mean(cdfs, axis=0)
        cdf_std = np.std(cdfs, axis=0)
        print(pdf_mean, pdf_std, cdf_mean, cdf_std)

        return pdf_mean, pdf_std, cdf_mean, cdf_std, bins

    def add_theoretical_distributions(fig, x, r, c, data_list, dist_type, row, col, showlegend):
        all_data = np.concatenate(data_list)

        # Chi-square distribution
        if dist_type == 'pdf':
            c = mean_data / stats.chi2.mean(df=r)
            y_chi = stats.chi2.pdf(x, df=r, scale=c)
        else:
            c = mean_data / stats.chi2.mean(df=r)
            y_chi = stats.chi2.cdf(x, df=r, scale=c)
        fig.add_trace(go.Scatter(x=x, y=y_chi, mode='lines', name='Chi-square', line=dict(color='red', dash='dash'),
                                 showlegend=showlegend),
                      row=row, col=col)

        if dist_type == 'pdf':
            c = mean_data / stats.chi.mean(df=r)
            y_chi = stats.chi.pdf(x, df=r, scale=c)
        else:
            c = mean_data / stats.chi.mean(df=r)
            y_chi = stats.chi.cdf(x, df=r, scale=c)
        fig.add_trace(go.Scatter(x=x, y=y_chi, mode='lines', name='Chi', line=dict(color='orange', dash='dot'),
                                 showlegend=showlegend), row=row, col=col)

        # Log-normal distribution
        if dist_type == 'pdf':
            c = mean_data / stats.gamma.mean(a=r)
            y_gamma = stats.gamma.pdf(x, a=r, scale=c)  # a是shape参数,等于自由度r
        else:
            c = mean_data / stats.gamma.mean(a=r)
            y_gamma = stats.gamma.cdf(x, a=r, scale=c)
        fig.add_trace(go.Scatter(x=x, y=y_gamma, mode='lines', name='Gamma',
                                 line=dict(color='green', dash='dot'), showlegend=showlegend), row=row, col=col)

        # Exponential distribution
        loc, scale = stats.expon.fit(all_data)
        if dist_type == 'pdf':
            y_expon = stats.expon.pdf(x, loc=loc, scale=scale)
        else:
            y_expon = stats.expon.cdf(x, loc=loc, scale=scale)
        fig.add_trace(
            go.Scatter(x=x, y=y_expon, mode='lines', name='Exponential', line=dict(color='purple', dash='dashdot'),
                       showlegend=showlegend),
            row=row, col=col)

    r = rank_to_reduce
    num_bins = int(2 * math.sqrt(num_samples))

    aug_list = {
        "num_graph": 1,
        "feature_dim": feature_dim,
        "num_layer": num_layer,
        "num_rank": rank_to_reduce,
        "model_type": model_type,
        "prompt_type": prompt_type,
        "graph_size_lo": 40,
        "graph_size_hi": 60,
        "num_to_loop": 1,
        "need_expect_value": False,
        "max_epochs": 2000,
        "experiment_type": "single_graph",
        "hyper_para": {
            "link_dense": 0.1,
            "trans_dense": 0.8,
            "prompt_num": 2,
            "token_num": 3,
            "inner_prune": 0.3,
            "cross_prune": 0.7,
            "cross_mat_num": 2,
        }
    }

    num_repeats = 1
    data_list = []
    for _ in tqdm(range(num_repeats), desc="Repeating Experiments", colour="blue"):
        data_curr = []
        for _ in tqdm(range(num_samples), desc="Processing", colour="green"):
            loss_curr = taskers.container(aug_list)[0]
            data_curr.append(loss_curr)
        data_list.append(np.array(data_curr))

    print(data_list)
    mean_data = np.mean([np.mean(data) for data in data_list])

    var_data = np.mean([np.var(data) for data in data_list])

    results = {
        "mean_loss": data_list,
        "variance": var_data
    }

    save_results_to_csv(results)
    pdf_mean, pdf_std, cdf_mean, cdf_std, bins = calculate_distributions(data_list, num_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    all_data = np.concatenate(data_list)
    observed_freq, _ = np.histogram(all_data, bins=bins)
    expected_freq = stats.chi.pdf(bin_centers, df=r, scale=c) * len(all_data) * (bins[1] - bins[0])
    print(observed_freq,expected_freq)

    chi2_stat, p_value = stats.chisquare(observed_freq, expected_freq)
    return


def experiment_3_docker(num_graph, num_layer, rank_to_reduce, feature_dim, need_ref, trans_dense, num_4_loss,
                        para_prompt, model_type, prompt_type):
    losses = []

    aug_list = {
        "num_graph": num_graph,
        "feature_dim": feature_dim,
        "num_layer": num_layer,
        "num_rank": rank_to_reduce,
        "model_type": model_type,
        "prompt_type": prompt_type,
        "graph_size_lo": 50,
        "graph_size_hi": 60,
        "num_to_loop": 1,
        "need_expect_value": need_ref,
        "max_epochs": 2000,
        "experiment_type": "multi_graph",
        "hyper_para": {
            "link_dense": 0.1,
            "trans_dense": trans_dense,
            "prompt_num": para_prompt,
            "token_num": para_prompt,
            "inner_prune": 0.3,
            "cross_prune": 0.7,
            "cross_mat_num": 4,
        }
    }

    if prompt_type == "gpf":

        loss = taskers.multi_experiment_4_lower_bound(aug_list)
        losses.append(loss)

    elif prompt_type == 'All-in-one' or 'All-in-one-plus':
        aug_list["num_to_loop"] = 1
        aug_list["max_epochs"] = aug_list["max_epochs"]

        loss = taskers.multi_experiment_4_quantile_bound(aug_list)
        losses.append(loss)

    losses = np.array(losses)

    (loss_mean, loss_var) = (losses.mean(), losses.var())

    print(
        f"loss_mean:{loss_mean}, loss_var:{loss_var}\n\n with para:[rank_to_reduce:{rank_to_reduce}, num_graph:{num_graph}]")

    return loss_mean, loss_var




if __name__ == "__main__":
    pass

