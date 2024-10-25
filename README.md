# NDDGNN

Directed Graph Neural Networks with Node Diversity


## Getting Started

To get up and running with the project, you need to first set up your environment and install the necessary dependencies. This guide will walk you through the process step by step.

### Setting Up the Environment

The project is designed to run on Python 3.10.

### Installing Dependencies

Once the environment is activated, install the required packages:

```bash
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg pytorch-sparse -c pyg
pip install ogb==1.3.6
pip install pytorch_lightning==2.0.2
pip install gdown==4.7.1
```


```bash
conda install pytorch==2.0.1 -c pytorch
```
## Dataset Fix
<font color='red'> 
For Citeseer-Full and Cora-ML datasets, PyG loads them as undirected by default. To utilize these datasets in their directed form, a slight modification is required in the PyG local installation. Please comment out the line `edge_index = to_undirected(edge_index, num_nodes=x.size(0))` in the file located at:

```bash
/miniconda3/envs/your_env/lib/python3.10/site-packages/torch_geometric/io/npz.py
```
</font>
## Running Experiments


We providee the node classification and link prediction running experiments here,

### NDDGNN Experiments

To reproduce the best NDDGNN results on node classification (Table 2 in our paper), use the following command:

```bash
python -m src.run --dataset cora_ml --use_best_hyperparams --num_runs 10
python -m src_large.run --dataset arxiv-year --use_best_hyperparams --num_runs 10
```
for more datasets, see ```bash bash.sh```

For the directed graph link prediction(Table 3 in our paper), use the following command:
```bash
python -m src.run_link --dataset chameleon --use_best_hyperparams --num_runs 10
```

The `--dataset` parameter specifies the dataset to be used. Replace `chameleon` with the name of the dataset you want to use.