import os
import numpy as np
import uuid

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint

from src.utils.utils import use_best_hyperparams, get_available_accelerator, use_Adi_best_hyperparams
from src.datasets.data_loading import get_dataset, get_dataset_split
from src.datasets.dataset import FullBatchGraphDataset
from src.model_Link import get_model, LightingFullBatchModelWrapper
from src.utils.arguments import args
from src.datasets.data_utils import mask_zero_in_out, degree_encoding
from src.utils.link_predict import task_define


def run(args):
    torch.manual_seed(0)
    # torch.set_float32_matmul_precision("highest")

    # Get dataset and dataloader
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = dataset._data
    data.mask = mask_zero_in_out(data.x, data.edge_index, is_plot=False)
    # data.ind_edge is prediction edge.


    val_accs, test_accs = [], []
    for num_run in range(args.num_runs):

        new_edge_index, data.ind_edge, data.y, train_mask, val_mask, test_mask = task_define(edge_index=data.edge_index,
                                                                                             seed=num_run)
        data.edge_index = new_edge_index

        data_loader = DataLoader(FullBatchGraphDataset(data), batch_size=1, collate_fn=lambda batch: batch[0])

        # Get model
        args.num_features, args.num_classes = data.num_features, 2
        deg_enc = degree_encoding(data.edge_index)
        model = get_model(args, mask=data.mask, deg_enc=deg_enc)
        lit_model = LightingFullBatchModelWrapper(
            model=model,
            lr=args.lr,
            weight_decay=args.weight_decay,
            evaluator=evaluator,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            beta=args.beta,
        )

        # Setup Pytorch Lighting Callbacks
        early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience)
        model_summary_callback = ModelSummary(max_depth=-1)
        if not os.path.exists(f"{args.checkpoint_directory}/"):
            os.mkdir(f"{args.checkpoint_directory}/")
        model_checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            dirpath=f"{args.checkpoint_directory}/{str(uuid.uuid4())}/",
        )

        # Setup Pytorch Lighting Trainer
        trainer = pl.Trainer(
            log_every_n_steps=1,
            max_epochs=args.num_epochs,
            callbacks=[
                early_stopping_callback,
                model_summary_callback,
                model_checkpoint_callback,
            ],
            profiler="simple" if args.profiler else None,
            accelerator=get_available_accelerator(),
            devices=[args.gpu_idx],
        )

        # Fit the model
        trainer.fit(model=lit_model, train_dataloaders=data_loader)

        # Compute validation and test accuracy
        val_acc = model_checkpoint_callback.best_model_score.item()
        test_acc = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]["test_acc"]
        test_accs.append(test_acc)
        val_accs.append(val_acc)

    print(f"Test Acc: {np.mean(test_accs)} +- {np.std(test_accs)}")


if __name__ == "__main__":
    args = use_Adi_best_hyperparams(args, args.dataset)
    run(args)
