"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
import argparse
from pathlib import Path
import random
import shutil

import torch

from utils import refine, save_tensor_to_image
from datasets import get_test_loader, get_filtered_chars, read_font, render
from models import Generator
from sconf import Config
from train import setup_transforms
from trainer import Metric


def create_img_files(
    val_dir="/home/server08/changhun_workspace/FFG-Korean/data/ttfs/val",
    out_dir="/home/server08/changhun_workspace/FFG-Korean/data/images/test",
    num_chars=10,
    overwrite=True
):
    char_filter = [chr(i) for i in range(int("AC00", 16), int("D7B0", 16))] # korean

    ttf_files = []
    for (dirpath, dirnames, filenames) in os.walk(val_dir):
        for filename in filenames:
            if filename.endswith(".ttf"):
                ttf_files.append(os.path.join(dirpath, filename))

    for ttf_file in ttf_files:
        ttf_filename = Path(ttf_file).stem
        if overwrite and os.path.exists(os.path.join(out_dir, ttf_filename)):
            shutil.rmtree(os.path.join(out_dir, ttf_filename), ignore_errors=True)
        os.makedirs(os.path.join(out_dir, ttf_filename), exist_ok=True)

        filtered_chars = list(set.intersection(set(get_filtered_chars(ttf_file)), set(char_filter)))
        selected_chars = random.sample(filtered_chars, num_chars)

        for char in selected_chars:
            img = render(read_font(ttf_file), char)
            filepath = os.path.join(*[out_dir, ttf_filename, f"{char}.png"])
            img.save(filepath)


def eval_metric(args, left_argv):
    cfg = Config(*args.config_paths, default="cfgs/defaults.yaml")
    cfg.argv_update(left_argv)
    img_dir = Path(args.result_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    _, val_transform = setup_transforms(cfg)

    g_kwargs = cfg.get('g_args', {})
    gen = Generator(1, cfg.C, 1, **g_kwargs).cuda()

    weight = torch.load(args.weight)
    if "generator_ema" in weight:
        weight = weight["generator_ema"]
    gen.load_state_dict(weight)
    gen.eval()

    metric = Metric()
    total_l1_loss, total_rmse, total_ssim, total_lpips, num_out = 0, 0, 0, 0, 0

    _, test_loader = get_test_loader(cfg, val_transform)
    for batch in test_loader:
        style_imgs = batch["style_imgs"].cuda()
        char_imgs = batch["source_imgs"].unsqueeze(1).cuda()
        target_imgs = batch["target_imgs"].cuda()

        out = gen.gen_from_style_char(style_imgs, char_imgs)
        out = refine(out)
        fonts = batch["fonts"]
        chars = batch["chars"]

        for image, font, char in zip(out, fonts, chars):
            (img_dir / font).mkdir(parents=True, exist_ok=True)
            path = img_dir / font / f"{char}.png"
            save_tensor_to_image(image, path)

        eval_metric = metric(target_imgs, out)

        total_l1_loss += sum(eval_metric["l1_loss"])
        total_rmse += sum(eval_metric["rmse"])
        total_ssim += sum(eval_metric["ssim"])
        total_lpips += sum(eval_metric["lpips"])
        num_out += len(style_imgs)

        del style_imgs, char_imgs, target_imgs, out

    l1_loss = total_l1_loss / num_out
    rmse = total_rmse / num_out
    ssim = total_ssim / num_out
    lpips = total_lpips / num_out

    print(f"Final Result\nl1_loss : {l1_loss}\nRMSE : {rmse}\nssim : {ssim}\nlpips : {lpips}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("config_paths", nargs="+", help="path to config.yaml")
    parser.add_argument("--weight", help="path to weight to evaluate.pth")
    parser.add_argument("--result_dir", help="path to save the result file")

    parser.add_argument("--option", type=int, default=2)
    parser.add_argument("--val_dir", default="")
    parser.add_argument("--out_dir", default="")

    args, left_argv = parser.parse_known_args()

    if args.option == 1:
        if args.val_dir and args.out_dir:
            create_img_files(val_dir=args.val_dir, out_dir=args.out_dir)
        else:
            create_img_files()
    elif args.option == 2:
        eval_metric(args, left_argv)