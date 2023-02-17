import colorsys
import enum
import math

# from .backbones.Transformer_GNN import Transformer_GNN
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any

import einops
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import pytorch_lightning as pl
import scipy
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn.models
import torchmetrics
import torchvision
import torchvision.transforms.functional as trF
from kornia.geometry.transform import Rotate as krot
from PIL import Image
from torch import Tensor
from torch.optim import Adam
from tqdm import tqdm
from transformers.optimization import Adafactor

import wandb

from . import backbones
from . import spatial_diffusion as sd
from . import spatial_diffusion_discrete as sdd

# import ark_TFConv, Eff_GAT, Eff_GAT_Discrete

matplotlib.use("agg")


def matrix_cumprod(matrixes, dim):
    cumprods = []

    cumprod = torch.eye(matrixes[0].shape[0])
    for matrix in matrixes:
        cumprod = cumprod @ matrix
        cumprods.append(cumprod)
    return cumprods


class GNN_Diffusion(sdd.GNN_Diffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Qs = []
        self.rot_K = 4

        for t in range(self.steps):
            beta_t = self.betas[t]
            Q_t = (1 - beta_t) * torch.eye(self.rot_K) + beta_t * torch.ones(
                (self.rot_K, self.rot_K)
            ) / self.rot_K
            Qs.append(Q_t)
        self.register_buffer(
            "overline_Q_rot", torch.stack(sdd.matrix_cumprod(torch.stack(Qs), 0))
        )

    def init_backbone(self):
        self.model = backbones.Eff_GAT_Discrete_ROT(
            steps=self.steps,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
        )

    def training_step(self, batch, batch_idx):
        batch_size = batch.batch.max().item() + 1
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()

        new_t = torch.gather(t, 0, batch.batch)

        losses = self.p_losses(
            batch.indexes % self.K,
            new_t,
            rot_start=batch.rot_index % self.rot_K,
            loss_type="huber",
            cond=batch.patches,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        if batch_idx == 0 and self.local_rank == 0:
            indexes = self.p_sample_loop(
                batch.indexes.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            index, pred_rot = indexes[-1]
            rots = torch.tensor(
                [
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                ]
            )

            save_path = Path(f"results/{self.logger.experiment.name}/train")
            for i in range(
                min(batch.batch.max().item(), 4)
            ):  # save max 4 images during training loop
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                gt_rot = batch.rot[idx]

                n_patches = batch.patches_dim[i].tolist()
                y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                real_grid = einops.rearrange(xy, "x y c-> (x y) c")
                pos = real_grid[index[idx]]
                rot = pred_rot[index[idx]]

                n_patches = batch.patches_dim[i]
                i_name = batch.ind_name[i]
                self.save_image_rotated(
                    patches_rgb=patches_rgb,
                    pos=pos,
                    gt_pos=gt_pos,
                    patches_dim=n_patches,
                    ind_name=i_name,
                    file_name=save_path,
                    gt_rotations=gt_rot,
                    pred_rotations=rots[rot],
                )
        self.log_dict(losses)

        return sum(l for l in losses.values())

    def p_losses(
        self,
        x_start,
        t,
        rot_start,
        noise=None,
        loss_type="l1",
        cond=None,
        edge_index=None,
        batch=None,
    ):
        x_start_one_hot = torch.nn.functional.one_hot(x_start)
        rot_start_one_hot = torch.nn.functional.one_hot(rot_start)

        x_noisy = self.q_sample(
            x_start=x_start_one_hot, t=t, overline_Q=self.overline_Q
        )

        rot_noisy = self.q_sample(
            x_start=rot_start_one_hot, t=t, overline_Q=self.overline_Q_rot
        )
        cond = rotate_images(cond, rot_noisy)

        patch_feats = self.visual_features(cond)
        batch_size = batch.max() + 1
        batch_one_hot = torch.nn.functional.one_hot(batch)
        prob = (
            batch_one_hot.float() @ torch.rand(batch_size, device=self.device)
            > self.classifier_free_prob
        )
        classifier_free_patch_feats = prob[:, None] * patch_feats

        x_prediction, rot_prediction = self.forward_with_feats(
            x_noisy,
            t,
            cond,
            edge_index,
            rot=rot_noisy,
            patch_feats=classifier_free_patch_feats,
            batch=batch,
        )

        x_loss = F.cross_entropy(x_prediction, x_start)
        rot_loss = F.cross_entropy(rot_prediction, rot_start)

        return {"x_loss": x_loss, "rot_loss": rot_loss}

    @torch.no_grad()
    def p_sample(
        self, x, rot, t, t_index, cond, edge_index, sampling_func, patch_feats, batch
    ):
        return sampling_func(x, rot, t, t_index, cond, edge_index, patch_feats, batch)

    @torch.no_grad()
    def p_sample_ddpm(self, x, rot, t, t_index, cond, edge_index, patch_feats, batch):
        prev_timestep = t - self.inference_ratio

        model_output_x, model_output_rot = self.forward_with_feats(
            x, t, cond, edge_index, rot=rot, patch_feats=patch_feats, batch=batch
        )

        # estimate x_0

        logits = torch.where(
            t[:, None].tile(model_output_x.shape[1]) == 0,
            model_output_x,
            self.q_posterior_logits(x, model_output_x, t, prev_timestep),
        )

        mask = (t != 0).reshape(x.shape[0], *([1] * (len(x.shape)))).to(logits.device)
        noise = torch.rand(logits.shape).to(logits.device)
        gumbel_noise = -torch.log(-torch.log(noise))
        x_sample = torch.argmax(logits + mask * gumbel_noise, -1)

        # estimate rot_0

        logits = torch.where(
            t[:, None].tile(model_output_rot.shape[1]) == 0,
            model_output_rot,
            self.q_posterior_logits(
                rot,
                model_output_rot,
                t,
                prev_timestep,
                K=self.rot_K,
                overline_Q=self.overline_Q_rot,
            ),
        )

        mask = (
            (t != 0).reshape(rot.shape[0], *([1] * (len(rot.shape)))).to(logits.device)
        )
        noise = torch.rand(logits.shape).to(logits.device)
        gumbel_noise = -torch.log(-torch.log(noise))
        rot_sample = torch.argmax(logits + mask * gumbel_noise, -1)

        return x_sample, rot_sample

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device

        b = shape[0]

        index = torch.randint(0, self.K, shape, device=device)
        rot = torch.randint(0, self.rot_K, shape, device=device)

        imgs = []

        patch_feats = self.visual_features(cond)

        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            index, rot = self.p_sample(
                index,
                rot,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                cond=cond,
                edge_index=edge_index,
                patch_feats=patch_feats,
                batch=batch,
            )

            imgs.append((index, rot))
        return imgs

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            preds = self.p_sample_loop(
                batch.indexes.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            pred_last_index = preds[-1]
            rots = torch.tensor(
                [
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                ]
            )

            for i in range(batch.batch.max() + 1):
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                gt_index = batch.indexes[idx] % self.K
                gt_rots = batch.rot[idx]

                pred_index, pred_rots = pred_last_index[idx]
                n_patches = batch.patches_dim[i].tolist()
                i_name = batch.ind_name[i]

                y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                real_grid = einops.rearrange(xy, "x y c-> (x y) c")
                pred_pos = real_grid[pred_index]

                correct = (pred_index == gt_index).all()
                rot_correct = pred_rots == gt_rots
                correct = correct and rot_correct

                if (
                    self.local_rank == 0
                    and batch_idx < 10
                    and i < min(batch.batch.max().item(), 4)
                ):
                    save_path = Path(f"results/{self.logger.experiment.name}/val")
                    self.save_image_rotated(
                        patches_rgb=patches_rgb,
                        pos=pred_pos,
                        gt_pos=gt_pos,
                        patches_dim=n_patches,
                        ind_name=i_name,
                        file_name=save_path,
                        correct=correct,
                        gt_rotations=gt_rots,
                        pred_rotations=rots[pred_rots],
                    )

                self.metrics[f"{tuple(n_patches)}_nImages"].update(1)
                self.metrics["overall_nImages"].update(1)
                if correct:
                    # if (assignement[:, 0] == assignement[:, 1]).all():
                    self.metrics[f"{tuple(n_patches)}_acc"].update(1)
                    self.metrics["overall_acc"].update(1)
                    # accuracy_dict[tuple(n_patches)].append(1)
                else:
                    self.metrics[f"{tuple(n_patches)}_acc"].update(0)
                    self.metrics["overall_acc"].update(0)
                    # accuracy_dict[tuple(n_patches)].append(0)

            self.log_dict(self.metrics)
        # return accuracy_dict


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def rotate_images(patches, rot_index):
    angles = (90 * rot_index).float()
    r = krot(angles, mode="nearest")
    rot2 = r(patches)
    return rot2
