import torch
import math
from torch_geometric.utils import degree

def graph_matching(trans_list, prompt_list):
    assert trans_list.shape == prompt_list.shape
    difference = torch.norm(trans_list - prompt_list, p=2, dim=1).mean()
    return difference

if __name__ == '__main__':
   pass