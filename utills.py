import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def plot_graph(data):
    # 将 PyG 数据转换为 NetworkX 图
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
    plt.title("Graph Visualization")
    plt.show()