import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn.functional import softmax
from torch.nn import ModuleList, Linear
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    GATConv,
    JumpingKnowledge,
)

from src.datasets.data_utils import get_norm_adj, tau_softmax


def get_conv(conv_type, input_dim, output_dim, alpha, mask, edge_index=None):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "nddgnn":
        return ADiGCNConv(input_dim, output_dim, alpha, mask, edge_index)
    else:
        raise ValueError(f"Convolution type {conv_type} not supported")

class ADiGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha, mask, deg_enc=None):
        super(ADiGCNConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin_src_to_dst = Linear(input_dim, output_dim)
        self.lin_dst_to_src = Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm, self.adj_t_norm = None, None

        self.mask = mask
        self.in_filter = Linear(input_dim, 1)
        self.out_filter = Linear(input_dim, 1)

        self.fc = Linear(input_dim, output_dim)

        self.in_degree = deg_enc[0].long()
        self.out_degree = deg_enc[1].long()

        self.in_deg_enc = nn.Embedding(int(max(self.in_degree) + 1), input_dim, padding_idx=0)
        self.out_deg_enc = nn.Embedding(int(max(self.out_degree) + 1), input_dim, padding_idx=0)

    def deg_filter(self, x, out_nei, in_nei, mask):
        C_out = self.out_filter((out_nei - x + self.out_deg_enc(self.out_degree)))  # N*1
        C_in = self.in_filter((in_nei - x + self.in_deg_enc(self.in_degree)))  # N*1

        # softmax, add tau
        C = tau_softmax(torch.cat((C_out, C_in), 1))
        C_out = C[:, 0].unsqueeze(1).float()
        C_in = C[:, 1].unsqueeze(1).float()


        C_out = torch.multiply(C_out, mask["out_deg_mask"].unsqueeze(1)) + mask["out_deg_mask_bias"].unsqueeze(1)
        C_in = torch.multiply(C_in, mask["in_deg_mask"].unsqueeze(1)) + mask["in_deg_mask_bias"].unsqueeze(1)


        return C_out, C_in  # N*1

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir")

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        out_nei = self.adj_norm @ x
        in_nei = self.adj_t_norm @ x

        C_out, C_in = self.deg_filter(x, out_nei, in_nei, self.mask)

        return torch.multiply(C_out, self.lin_src_to_dst(out_nei)) + torch.multiply(C_in,
        self.lin_dst_to_src(in_nei)) + self.alpha * self.fc(x), [C_in, C_out]


class GNN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dim,
        num_layers=2,
        dropout=0,
        conv_type="dir-gcn",
        jumping_knowledge=False,
        normalize=False,
        alpha=1 / 2,
        learn_alpha=False,
        mask=None,
        deg_enc=None,
    ):
        super(GNN, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)
        output_dim = hidden_dim if jumping_knowledge else num_classes
        if num_layers == 1:
            self.convs = ModuleList([get_conv(conv_type, num_features, output_dim, self.alpha, mask, deg_enc)])
        else:
            self.convs = ModuleList([get_conv(conv_type, num_features, hidden_dim, self.alpha, mask, deg_enc)])
            for _ in range(num_layers - 2):
                self.convs.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha, mask, deg_enc))
            self.convs.append(get_conv(conv_type, hidden_dim, output_dim, self.alpha, mask, deg_enc))

        if jumping_knowledge is not None:
            input_dim = hidden_dim * num_layers if jumping_knowledge == "cat" else hidden_dim
            self.lin = Linear(input_dim*2, num_classes)
            self.jump = JumpingKnowledge(mode=jumping_knowledge, channels=hidden_dim, num_layers=num_layers)

        self.num_layers = num_layers
        self.dropout = dropout
        self.jumping_knowledge = jumping_knowledge
        self.normalize = normalize

    def forward(self, x, edge_index, ind_edge):
        xs = []
        for i, conv in enumerate(self.convs):
            x, [C_in, C_out] = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jumping_knowledge:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]
            if i == 0:
                C_ins = C_in
                C_outs = C_out
            else:
                C_ins = C_ins + C_in
                C_outs = C_outs + C_out

        C_ins = C_ins / self.num_layers
        C_outs = C_outs / self.num_layers
        if self.jumping_knowledge is not None:
            x = self.jump(xs)

            x = torch.cat((x[ind_edge[0, :]], x[ind_edge[1, :]]), axis=-1)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1), C_ins, C_outs


class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None, beta=0.5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask
        self.beta = beta

    def training_step(self, batch, batch_idx):
        x, y, edge_index, ind_edge = batch.x, batch.y.long(), batch.edge_index, batch.ind_edge
        out, C_ins, C_outs = self.model(x, edge_index, ind_edge)

        loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        loss_dis = self.distance_loss(C_ins, C_outs)
        loss = loss + self.beta*loss_dis
        self.log("train_loss", loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        self.log("train_acc", train_acc)
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc)

        return loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]

        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index, ind_edge = batch.x, batch.y.long(), batch.edge_index, batch.ind_edge
        out = self.model(x, edge_index, ind_edge)

        y_pred = out[0].max(1)[1]
        val_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", val_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def distance_loss(self, C_ins, C_outs):
        return (torch.sum(torch.pow((C_ins-C_ins.mean()), 2)) + torch.sum(torch.pow((C_outs-C_outs.mean()), 2)))

def get_model(args, mask, deg_enc=None):
    return GNN(
        num_features=args.num_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        conv_type=args.conv_type,
        jumping_knowledge=args.jk,
        normalize=args.normalize,
        alpha=args.alpha,
        learn_alpha=args.learn_alpha,
        mask = mask,
        deg_enc = deg_enc
    )
