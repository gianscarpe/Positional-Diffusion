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

# import ark_TFConv, Eff_GAT, Eff_GAT_Discrete

matplotlib.use("agg")


def matrix_cumprod(matrixes, dim):
    cumprods = []

    cumprod = torch.eye(matrixes[0].shape[0])
    for matrix in matrixes:
        cumprod = cumprod @ matrix
        cumprods.append(cumprod)
    return cumprods


class GNN_Diffusion(sd.GNN_Diffusion):
    def __init__(self, puzzle_sizes, *args, **kwargs):
        K = puzzle_sizes[0][0] * puzzle_sizes[0][1]
        kwargs["sampling"] = "DDPM"
        super().__init__(
            input_channels=K,
            output_channels=K,
            scheduler=sd.ModelScheduler.COSINE_DISCRETE,
            *args,
            **kwargs,
        )
        self.puzzle_sizes = puzzle_sizes[0]
        self.K = K
        Qs = []

        for t in range(self.steps):
            beta_t = self.betas[t]
            Q_t = (1 - beta_t) * torch.eye(self.K) + beta_t * torch.ones(
                (self.K, self.K)
            ) / self.K
            Qs.append(Q_t)

        self.register_buffer(
            "overline_Q", torch.stack(matrix_cumprod(torch.stack(Qs), 0))
        )

    def init_backbone(self):
        self.model = backbones.Eff_GAT_Discrete(
            steps=self.steps,
            input_channels=self.input_channels,
            output_channels=self.output_channels,
        )

    def training_step(self, batch, batch_idx):
        # return super().training_step(*args, **kwargs)
        batch_size = batch.batch.max().item() + 1
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()

        new_t = torch.gather(t, 0, batch.batch)

        loss = self.p_losses(
            batch.indexes % self.K,
            new_t,
            loss_type="huber",
            cond=batch.patches,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        if batch_idx == 0 and self.local_rank == 0:
            indexes = self.p_sample_loop(
                batch.indexes.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            index = indexes[-1]

            save_path = Path(f"results/{self.logger.experiment.name}/train")
            for i in range(
                min(batch.batch.max().item(), 4)
            ):  # save max 4 images during training loop
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                n_patches = batch.patches_dim[i].tolist()
                y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                real_grid = einops.rearrange(xy, "x y c-> (x y) c")
                pos = real_grid[index[idx]]

                n_patches = batch.patches_dim[i]
                i_name = batch.ind_name[i]
                if self.rotation:
                    gt_pos = batch.x[idx, :2]
                    pos = index[idx, :2]
                    pred_rot = index[idx, 2:]
                    gt_rot = batch.x[idx, 2:]
                    self.save_image_rotated(
                        patches_rgb=patches_rgb,
                        pos=pos,
                        gt_pos=gt_pos,
                        patches_dim=n_patches,
                        ind_name=i_name,
                        file_name=save_path,
                        gt_rotations=gt_rot,
                        pred_rotations=pred_rot,
                    )
                else:
                    self.save_image(
                        patches_rgb=patches_rgb,
                        pos=pos,
                        gt_pos=gt_pos,
                        patches_dim=n_patches,
                        ind_name=i_name,
                        file_name=save_path,
                    )

        self.log("loss", loss)

        return loss

    # forward diffusion
    def q_sample(self, x_start, t):
        Q_t = self.overline_Q[t]
        result = torch.bmm(x_start.float().unsqueeze(1), Q_t)

        return result.squeeze()

    def q_sample_reverse(self, x, x_start_estimated, t, previous_t):
        Q_t = self.overline_Q[t]
        Q_previous_t = self.overline_Q[previous_t]

        num = (
            torch.bmm(F.one_hot(x, self.K).float().unsqueeze(1), Q_t)
            * torch.bmm(
                F.one_hot(x_start_estimated, self.K).float().unsqueeze(1),
                Q_previous_t.transpose(2, 1),
            )
        ).squeeze()

        q_x = torch.bmm(Q_t, F.one_hot(x, num_classes=self.K).unsqueeze(-1).float())
        den = F.one_hot(x_start_estimated, self.K).float().unsqueeze(1) @ q_x
        return num / den.squeeze().unsqueeze(1)

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l1",
        cond=None,
        edge_index=None,
        batch=None,
    ):
        x_start_one_hot = torch.nn.functional.one_hot(x_start)

        x_noisy_prob = self.q_sample(x_start=x_start_one_hot, t=t)
        x_noisy = torch.distributions.categorical.Categorical(x_noisy_prob).sample()

        if self.rotation:
            cond = rotate_images(cond, x_noisy[:, -2:])

        patch_feats = self.visual_features(cond)
        batch_size = batch.max() + 1
        batch_one_hot = torch.nn.functional.one_hot(batch)
        prob = (
            batch_one_hot.float() @ torch.rand(batch_size, device=self.device)
            > self.classifier_free_prob
        )
        classifier_free_patch_feats = prob[:, None] * patch_feats

        prediction = self.forward_with_feats(
            x_noisy,
            t,
            cond,
            edge_index,
            patch_feats=classifier_free_patch_feats,
            batch=batch,
        )
        loss = F.cross_entropy(prediction, x_start)

        return loss

    @torch.no_grad()
    def p_sample(
        self, x, t, t_index, cond, edge_index, sampling_func, patch_feats, batch
    ):
        return sampling_func(x, t, t_index, cond, edge_index, patch_feats, batch)

    @torch.no_grad()
    def p_sample_ddpm(self, x, t, t_index, cond, edge_index, patch_feats, batch):
        prev_timestep = t - self.inference_ratio

        if self.classifier_free_prob > 0.0:
            model_output_cond = self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )

            model_output_uncond = self.forward_with_feats(
                x,
                t,
                cond,
                edge_index,
                patch_feats=torch.zeros_like(patch_feats),
                batch=batch,
            )
            model_output = (
                1 + self.classifier_free_w
            ) * model_output_cond - self.classifier_free_w * model_output_uncond
        else:
            model_output = self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )

        # estimate x_0

        x_start_estimated_prob = F.softmax(model_output)

        x_start_estimated = x_start_estimated_prob.argmax(1)

        logits = self.q_sample_reverse(x, x_start_estimated, t, prev_timestep)

        p_tilde = logits * x_start_estimated_prob

        prev_sample = torch.distributions.Categorical(p_tilde).sample()

        return prev_sample

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device

        b = shape[0]

        index = torch.randint(0, self.K, shape, device=device)

        imgs = []

        patch_feats = self.visual_features(cond)

        # time_t = torch.full((b,), i, device=device, dtype=torch.long)

        # time_t = torch.full((b,), 0, device=device, dtype=torch.long)

        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            index = self.p_sample(
                index,
                torch.full((b,), i, device=device, dtype=torch.long),
                # time_t + i,
                i,
                cond=cond,
                edge_index=edge_index,
                patch_feats=patch_feats,
                batch=batch,
            )

        imgs.append(index)
        return imgs

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            pred_indexes = self.p_sample_loop(
                batch.indexes.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            pred_last_index = pred_indexes[-1]

            for i in range(batch.batch.max() + 1):
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                gt_index = batch.indexes[idx]
                pred_index = pred_last_index[idx]
                n_patches = batch.patches_dim[i].tolist()
                i_name = batch.ind_name[i]

                y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                real_grid = einops.rearrange(xy, "x y c-> (x y) c")
                pred_pos = real_grid[pred_index]

                correct = (pred_index == gt_index).all()
                if self.rotation:
                    pred_rot = indexes[idx, 2:]
                    gt_rot = batch.x[idx, 2:]
                    rot_correct = (
                        torch.cosine_similarity(pred_rot, gt_rot)
                        > math.cos(math.pi / 4)
                    ).all()
                    correct = correct and rot_correct

                if (
                    self.local_rank == 0
                    and batch_idx < 10
                    and i < min(batch.batch.max().item(), 4)
                ):
                    save_path = Path(f"results/{self.logger.experiment.name}/val")

                    if self.rotation:
                        self.save_image_rotated(
                            patches_rgb=patches_rgb,
                            pos=pos,
                            gt_pos=gt_pos,
                            patches_dim=n_patches,
                            ind_name=i_name,
                            file_name=save_path,
                            correct=correct,
                            gt_rotations=gt_rot,
                            pred_rotations=pred_rot,
                        )
                    else:
                        self.save_image(
                            patches_rgb=patches_rgb,
                            pos=pred_pos,
                            gt_pos=gt_pos,
                            patches_dim=n_patches,
                            ind_name=i_name,
                            file_name=save_path,
                            correct=correct,
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
