import numpy
import random
import torch
from torch_geometric.utils import negative_sampling


def task_define(edge_index, train_size=0.85, val_size=0.05,test_size=0.1, seed=None):
    torch.manual_seed(seed)
    random.seed(seed)

    e = len(edge_index[0])
    train_num = int(train_size*e)
    val_num = int(val_size * e)
    test_num = int(test_size * e)
    e_index = [i for i in range(e)]

    test_mask = random.sample(e_index, test_num)
    test_edge = edge_index[:, test_mask]

    # remove test edge
    new_edge_index = edge_index[:, list(set(e_index)-set(test_mask))]

    train_mask = random.sample(list(set(e_index)-set(test_mask)), train_num)
    train_edge = edge_index[:, train_mask]

    val_mask = list(set(e_index)-set(test_mask) - set(train_mask))
    val_edge = edge_index[:, val_mask]

    # generate negative edge
    neg_train_edge, neg_val_edge, neg_test_edge = random_sample_neg_edge(edge_index, train_num, val_edge.size(1), test_num)

    train_edge = torch.cat((train_edge, neg_train_edge), 1)
    val_edge = torch.cat((val_edge, neg_val_edge), 1)
    test_edge = torch.cat((test_edge, neg_test_edge), 1)

    ind_edge = torch.cat((train_edge, val_edge, test_edge), 1)

    train_mask = torch.tensor([True for i in range(train_edge.size(1))] + [False for i in range(val_edge.size(1))] + [False for i in range(test_edge.size(1))])

    val_mask = torch.tensor(
        [False for i in range(train_edge.size(1))] + [True for i in range(val_edge.size(1))] + [False for i in range(
            test_edge.size(1))])
    train_mask = torch.tensor(
        [False for i in range(train_edge.size(1))] + [False for i in range(val_edge.size(1))] + [True for i in range(
            test_edge.size(1))])



    y_train = torch.cat((torch.ones((neg_train_edge.size(1), 1)), torch.zeros((neg_train_edge.size(1), 1))), 0)
    y_val = torch.cat((torch.ones((neg_val_edge.size(1), 1)), torch.zeros((neg_val_edge.size(1), 1))), 0)
    y_test = torch.cat((torch.ones((neg_test_edge.size(1), 1)), torch.zeros((neg_test_edge.size(1), 1))), 0)
    y = torch.cat((y_train,y_val,y_test), 0)

    return new_edge_index, ind_edge, y, train_mask, val_mask, test_mask

def random_sample_neg_edge(edge_index, train_num, val_num, test_num):
    neg_edge_index = negative_sampling(
        edge_index=edge_index,
        num_nodes=max(edge_index[0]),
        num_neg_samples=edge_index.size(1))
    neg_train_edge = neg_edge_index[:, :train_num]
    neg_val_edge = neg_edge_index[:, train_num: train_num+val_num]
    neg_test_edge = neg_edge_index[:, train_num+val_num:]
    return neg_train_edge, neg_val_edge, neg_test_edge