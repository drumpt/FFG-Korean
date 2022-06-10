"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import torch
import torch.nn as nn
import numpy as np

from IQA_pytorch import SSIM
import lpips

import utils


def torch_eval(val_fn):
    @torch.no_grad()
    def decorated(self, gen, *args, **kwargs):
        gen.eval()
        ret = val_fn(self, gen, *args, **kwargs)
        gen.train()

        return ret

    return decorated


class Evaluator:
    def __init__(self, writer):
        torch.backends.cudnn.benchmark = True
        self.writer = writer

    @torch_eval
    def comparable_val_saveimg(self, gen, loader, step, n_row, tag='val'):
        compare_batches = self.infer_fact_loader(gen, loader)
        comparable_grid = utils.make_comparable_grid(*compare_batches[::-1], nrow=n_row)
        saved_path = self.writer.add_image(tag, comparable_grid, global_step=step)

        return comparable_grid, saved_path

    @torch_eval
    def infer_fact_loader(self, gen, loader, save_dir=None):
        outs = []
        trgs = []

        for batch in loader:
            style_imgs = batch["style_imgs"].cuda()
            char_imgs = batch["source_imgs"].unsqueeze(1).cuda()

            out = gen.gen_from_style_char(style_imgs, char_imgs)
            outs.append(out.detach().cpu())
            if "trg_imgs" in batch:
                trgs.append(batch["trg_imgs"])

        outs = torch.cat(outs).float()
        ret = (outs,)
        if trgs:
            trgs = torch.cat(trgs)
            ret += (trgs,)

        return ret


class Metric(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.ssim = SSIM(channels=3)
        self.lpips = lpips.LPIPS(net='alex')

    def forward(self, gt_batch, gen_batch):
        l1_loss_batch, rmse_batch, lpips_batch = [], [], []

        # convert grayscale to RGB image
        if gt_batch.shape[1] == 1:
            gt_batch = torch.cat([gt_batch for _ in range(3)], dim=1)
        if gen_batch.shape[1] == 1:
            gen_batch = torch.cat([gen_batch for _ in range(3)], dim=1)

        for gt_img, gen_img in zip(gt_batch, gen_batch):
            l1_loss = self.l1(gt_img, gen_img)
            rmse = torch.sqrt(self.mse(gt_img, gen_img))
            lpips = self.lpips((2 * gt_img - 1).cpu(), (2 * gen_img - 1).cpu())

            l1_loss_batch.append(l1_loss.item())
            rmse_batch.append(rmse.item())
            lpips_batch.append(lpips.item())

        ssim_batch = self.ssim(gt_batch, gen_batch, as_loss=False).tolist()

        result = {
            "l1_loss": l1_loss_batch,
            "rmse": rmse_batch,
            "ssim": ssim_batch,
            "lpips": lpips_batch
        }

        return result
