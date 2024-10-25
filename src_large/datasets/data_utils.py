import torch
from torch_sparse import mul
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch_geometric
from torch_sparse import SparseTensor
import torch
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def row_norm(adj):
    """
    Applies the row-wise normalization:
        \mathbf{D}_{out}^{-1} \mathbf{A}
    """
    row_sum = sparsesum(adj, dim=1)

    return mul(adj, 1 / row_sum.view(-1, 1))


def directed_norm(adj):
    """
    Applies the normalization for directed graphs:
        \mathbf{D}_{out}^{-1/2} \mathbf{A} \mathbf{D}_{in}^{-1/2}.
    """

    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1) # 1
    out_deg_inv_sqrt = out_deg.pow_(-0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj

def directed_C_norm(adj):
    """
        Applies the normalization for directed graphs:
            \mathbf{D}_{out}^{1/2} \mathbf{A} \mathbf{D}_{in}^{1/2}.
        """
    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(0.5)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=1)  # 1
    out_deg_inv_sqrt = out_deg.pow_(0.5)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))
    return adj


def all_norm(x, edge_index):
    row, col = edge_index
    num_nodes = x.shape[0]
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))

    in_deg = sparsesum(adj, dim=0)
    in_deg_inv_sqrt = in_deg.pow_(-0.25)
    in_deg_inv_sqrt.masked_fill_(in_deg_inv_sqrt == float("inf"), 0.0)

    in_deg_t = sparsesum(adj_t, dim=0)
    in_deg_inv_sqrt_t = in_deg_t.pow_(-0.25)
    in_deg_inv_sqrt_t.masked_fill_(in_deg_inv_sqrt_t == float("inf"), 0.0)

    out_deg = sparsesum(adj, dim=0)
    out_deg_inv_sqrt = out_deg.pow_(-0.25)
    out_deg_inv_sqrt.masked_fill_(out_deg_inv_sqrt == float("inf"), 0.0)

    out_deg_t = sparsesum(adj_t, dim=0)
    out_deg_inv_sqrt_t = out_deg_t.pow_(-0.25)
    out_deg_inv_sqrt_t.masked_fill_(out_deg_inv_sqrt_t == float("inf"), 0.0)

    # 继续写
    out = mul()
    adj = mul(adj, out_deg_inv_sqrt.view(-1, 1))
    adj = mul(adj, in_deg_inv_sqrt.view(1, -1))

    return adj
def get_norm_adj(adj, norm):
    if norm == "sym":
        return gcn_norm(adj, add_self_loops=False)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    elif norm == "dir-C":
        return directed_C_norm(adj)
    else:
        raise ValueError(f"{norm} normalization is not supported")


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


def get_adj(edge_index, num_nodes, graph_type="directed"):
    """
    Return the type of adjacency matrix specified by `graph_type` as sparse tensor.
    """
    if graph_type == "transpose":
        edge_index = torch.stack([edge_index[1], edge_index[0]])
    elif graph_type == "undirected":
        edge_index = torch_geometric.utils.to_undirected(edge_index)
    elif graph_type == "directed":
        pass
    else:
        raise ValueError(f"{graph_type} is not a valid graph type")

    value = torch.ones((edge_index.size(1),), device=edge_index.device)
    return SparseTensor(row=edge_index[0], col=edge_index[1], value=value, sparse_sizes=(num_nodes, num_nodes))


def compute_unidirectional_edges_ratio(edge_index):
    num_directed_edges = edge_index.shape[1]
    num_undirected_edges = torch_geometric.utils.to_undirected(edge_index).shape[1]

    num_unidirectional = num_undirected_edges - num_directed_edges

    return (num_unidirectional / (num_undirected_edges / 2)) * 100

def mask_zero_in_out(x, edge_index, is_plot=False):
    '''
    if node i without indegree or outdegree, we mask it，
    C_in_{i} = 0, C_out_{i} = 1, if d_in_{i} = 0

    output: in_deg_mask, out_deg_mask, in_deg_mask_bias, out_deg_mask_bias

    eg: a1 = in_deg_mask, b1 = in_deg_mask_bias, a2 = out_deg_mask, b2 = out_deg_mask_bias
    Mask operation
    C1 = C1.multiply(a1) + b1
    C2 = C2.multiply(a1) + b2
    '''
    row, col = edge_index
    num_nodes = x.shape[0]

    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    in_deg = sparsesum(adj, dim=0)
    out_deg = sparsesum(adj, dim=1)
    if is_plot:
        plot_in_out_degre_distribution(in_deg.detach().numpy().tolist(), out_deg.detach().numpy().tolist())
    in_deg_mask = torch.ones(x.shape[0])
    out_deg_mask = torch.ones(x.shape[0])
    in_deg_mask_bias = torch.zeros(x.shape[0])
    out_deg_mask_bias = torch.zeros(x.shape[0])
    for i in range(len(in_deg)):
        if in_deg[i] == 0 or (in_deg[i] == 1 and adj[i, i] == 1):
            in_deg_mask[i] = 0
            out_deg_mask[i] = 0
            out_deg_mask_bias[i] = 1
        if out_deg[i] == 0:
            in_deg_mask[i] = 0
            out_deg_mask[i] = 0
            in_deg_mask_bias[i] = 1

    return {"in_deg_mask":in_deg_mask.to(device),
            "out_deg_mask":out_deg_mask.to(device),
            "in_deg_mask_bias":in_deg_mask_bias.to(device),
            "out_deg_mask_bias":out_deg_mask_bias.to(device)}

def plot_in_out_degre_distribution(in_d, out_d):
    '''
    in_d, out_d : list
    '''

    max_d = max(max(in_d), max(out_d))
    in_d = Counter(in_d)
    out_d = Counter(out_d)
    in_d_dis = []
    out_d_dis = []
    for i in range(int(max_d)):
        if i in in_d.keys():
            in_d_dis.append(in_d[i])
        else:
            in_d_dis.append(0)
        if i in out_d.keys():
            out_d_dis.append(out_d[i])
        else:
            out_d_dis.append(0)
    plt.bar(range(len(in_d_dis)), in_d_dis, label='In degree', alpha=0.7,
            color='lightcoral')  # alpha是透明度
    plt.bar(range(len(out_d_dis)), out_d_dis, label='Out degree', alpha=0.5, color='dodgerblue')
    # plt.gca().set_aspect(1)
    # plt.plot(np.arange(0, 20), np.arange(0.025, 1.025, 0.05), ls='--', c='k')
    # plt.xticks(range(0, 21, 4), [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.xlim(0, 20)
    # plt.ylim(0, 1.0)
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Number', fontsize=14)
    plt.legend(fontsize=14)
    # plt.title(args.data + ',' + str(args.labelrate) + ',' + args.model, fontsize=16, fontweight="bold")

    plt.show()

def tau_softmax(input, tau=1):
    input_exp = torch.exp(input/tau)
    partition = input_exp.sum(1, keepdim=True)
    return input_exp / partition

def degree_encoding(edge_index):
    row, col = edge_index
    num_nodes = max(max(row), max(col)) + 1
    adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    in_deg_enconde = sparsesum(adj, dim=0)
    out_deg_enconde = sparsesum(adj, dim=1)
    return [in_deg_enconde.to(device), out_deg_enconde.to(device)]

def plot_bi_filter(f_in, f_out):
    f_in = f_in.detach().cpu().numpy().tolist()
    f_out = f_out.detach().cpu().numpy().tolist()
    plt.plot(range(len(f_in)), f_in, label='In_filter')
    plt.plot(range(len(f_out)), f_out, label='Out_filter')
    plt.legend()
    plt.show()

def row_l1_normalize(X):
    norm = 1e-6 + X.sum(dim=1, keepdim=True)
    return X/norm