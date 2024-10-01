import experiment


def my_experiment():
    model_list = ["GCN", "GAT"]
    prompt_list = ["gpf", "gpf-plus", "All-in-one", "All-in-one-plus"]
    experiment.experiment_1_1(model_type="GCN", prompt_type="gpf")

if __name__ == "__main__":
    my_experiment()