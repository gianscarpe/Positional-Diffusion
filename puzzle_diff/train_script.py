import argparse
import os
import sys

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import math
import random
import string

import matplotlib
import pytorch_lightning as pl
from dataset import dataset_utils as du
from model import spatial_diffusion as sd
from model import spatial_diffusion_discrete as sdd
from model import spatial_diffusion_discrete_rot as sdd_rot
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb

matplotlib.use("qtagg")


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str  # print("Random string of length", length, "is:", result_str)


def main(
    batch_size,
    gpus,
    steps,
    num_workers,
    dataset,
    puzzle_sizes,
    sampling,
    inference_ratio,
    offline,
    classifier_free_prob,
    classifier_free_w,
    noise_weight,
    data_augmentation,
    checkpoint_path,
    rotation,
    predict_xstart,
    discrete,
    loss_type,
    only_rotation,
    cold_diffusion,
):
    ### Define dataset

    if rotation:
        train_dt, test_dt, puzzle_sizes = du.get_dataset_ROT(
            dataset=dataset,
            puzzle_sizes=puzzle_sizes,
            augment=data_augmentation,
        )
    else:
        train_dt, test_dt, puzzle_sizes = du.get_dataset(
            dataset=dataset, puzzle_sizes=puzzle_sizes, augment=data_augmentation
        )

    dl_train = torch_geometric.loader.DataLoader(
        train_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    if discrete and rotation:
        model = sdd_rot.GNN_Diffusion(
            steps=steps,
            sampling=sampling,
            inference_ratio=inference_ratio,
            classifier_free_w=classifier_free_w,
            classifier_free_prob=classifier_free_prob,
            noise_weight=noise_weight,
            rotation=rotation,
            model_mean_type=sd.ModelMeanType.START_X,
            puzzle_sizes=puzzle_sizes,
            scheduler=sd.ModelScheduler.LINEAR,
            loss_type=loss_type,
            only_rotation=only_rotation,
            cold_diffusion=cold_diffusion,
        )
    elif discrete:
        model = sdd.GNN_Diffusion(
            steps=steps,
            sampling=sampling,
            inference_ratio=inference_ratio,
            classifier_free_w=classifier_free_w,
            classifier_free_prob=classifier_free_prob,
            noise_weight=noise_weight,
            rotation=rotation,
            model_mean_type=sd.ModelMeanType.START_X,
            puzzle_sizes=puzzle_sizes,
            scheduler=sd.ModelScheduler.LINEAR,
            loss_type=loss_type,
        )
    else:
        model = sd.GNN_Diffusion(
            steps=steps,
            sampling=sampling,
            inference_ratio=inference_ratio,
            classifier_free_w=classifier_free_w,
            classifier_free_prob=classifier_free_prob,
            noise_weight=noise_weight,
            rotation=rotation,
            model_mean_type=sd.ModelMeanType.EPISLON
            if not predict_xstart
            else sd.ModelMeanType.START_X,
        )

    model.initialize_torchmetrics(puzzle_sizes)

    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"{dataset}-{puzzle_sizes}-{steps}-{get_random_string(6)}-{'discrete' if discrete else ''}"

    if rotation:
        experiment_name = "ROT-" + experiment_name

    tags = [f"{dataset}", f'{"franklin" if franklin else "fisso"}', "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=offline,
        name=experiment_name,
        # entity="puzzle_diff",
        entity="puzzle_diff_academic",
        tags=tags,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="overall_acc", mode="max", save_top_k=2
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
    )

    trainer.fit(model, dl_train, dl_test, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=6)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=300)
    ap.add_argument("-num_workers", type=int, default=8)
    ap.add_argument(
        "-dataset", default="wikiart", choices=["celeba", "wikiart", "cifar100"]
    )
    ap.add_argument("-sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("-inference_ratio", type=int, default=10)
    ap.add_argument(
        "-puzzle_sizes", nargs="+", default=[6], type=int, help="Input a list of values"
    )
    ap.add_argument("--offline", action="store_true", default=False)
    ap.add_argument("--classifier_free_w", type=float, default=0.2)
    ap.add_argument("--classifier_free_prob", type=float, default=0.0)
    ap.add_argument("--data_augmentation", type=str, default="none")
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--noise_weight", type=float, default=0.0)
    ap.add_argument("--predict_xstart", type=bool, default=False)
    ap.add_argument("--rotation", type=bool, default=False)
    ap.add_argument("--only_rotation", type=bool, default=False)
    ap.add_argument("--discrete", type=bool, default=False)
    ap.add_argument("--cold_diffusion", type=bool, default=False)
    ap.add_argument("--loss_type", type=str, default="cross_entropy")
    args = ap.parse_args()
    print(args)
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
        dataset=args.dataset,
        puzzle_sizes=args.puzzle_sizes,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        offline=args.offline,
        classifier_free_prob=args.classifier_free_prob,
        classifier_free_w=args.classifier_free_w,
        noise_weight=args.noise_weight,
        data_augmentation=args.data_augmentation,
        checkpoint_path=args.checkpoint_path,
        rotation=args.rotation,
        predict_xstart=args.predict_xstart,
        discrete=args.discrete,
        loss_type=args.loss_type,
        only_rotation=args.only_rotation,
        cold_diffusion=args.cold_diffusion,
    )
