"""
MX-Font
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

import os
from pathlib import Path
import json
from PIL import Image

import torch
from torch.utils.data import Dataset

from .ttf_utils import get_filtered_chars, read_font, render


class ImageTestDataset(Dataset):
    def __init__(self, data_dir, source_font, gen_chars_file=None, transform=None, extension="png"):
        self.data_dir = Path(data_dir)
        self.source_font = read_font(source_font)
        self.gen_chars = get_filtered_chars(source_font)

        self.source_font_dir = source_font
        self.use_target_file = use_target_file

        if gen_chars_file is not None:
            gen_chars = json.load(open(gen_chars_file))
            self.gen_chars = list(set(self.gen_chars).intersection(set(gen_chars)))

        self.font_ref_chars = self.load_data_list(self.data_dir, extension)

        self.gen_char_dict = {k: self.gen_chars for k in self.font_ref_chars}
        self.data_list = [(key, char) for key, chars in self.gen_char_dict.items() for char in chars]
        self.transform = transform

    def load_data_list(self, data_dir, extension):
        fonts = [x.name for x in data_dir.iterdir() if x.is_dir()]

        font_chars = {}
        for font in fonts:
            chars = [x.name for x in (self.data_dir / font).glob(f"*.{extension}")]
            font_chars[font] = chars
        return font_chars

    def __getitem__(self, index):
        font, char = self.data_list[index]
        ref_imgs = torch.stack([self.transform(Image.open(str(self.data_dir / font / f"{rc}")))
                                for rc in self.font_ref_chars[font]])
        source_img = self.transform(render(self.source_font, char))

        ret = {
            "style_imgs": ref_imgs,
            "source_imgs": source_img,
            "fonts": font,
            "chars": char,
        }

        if self.use_target_file:
            ttf_filename = Path(self.source_font_dir).stem
            target_font_dir = self.source_font_dir.replace(ttf_filename, font)
            target_font = read_font(target_font_dir)

            target_img = self.transform(render(target_font, char))
            ret["target_imgs"] = target_img

        return ret

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "source_imgs": torch.stack(_ret["source_imgs"]),
            "target_imgs": torch.stack(_ret["target_imgs"]),
            "fonts": _ret["fonts"],
            "chars": _ret["chars"],
        }

        return ret


class ImageEvalDataset(Dataset):
    def __init__(self, data_dir, source_font, target_dir, gen_chars_file=None, transform=None, extension="png"):
        self.data_dir = Path(data_dir)
        self.source_font = read_font(source_font)
        self.gen_chars = get_filtered_chars(source_font)

        if gen_chars_file is not None:
            gen_chars = json.load(open(gen_chars_file))
            self.gen_chars = list(set(self.gen_chars).intersection(set(gen_chars)))

        self.font_ref_chars = self.load_data_list(self.data_dir, extension)
        self.target_font_dit = self.load_target_font_dict(target_dir, self.font_ref_chars.keys())

        self.gen_char_dict = {k: self.gen_chars for k in self.font_ref_chars}
        self.data_list = [(key, char) for key, chars in self.gen_char_dict.items() for char in chars]
        self.transform = transform

    def load_data_list(self, data_dir, extension):
        fonts = [x.name for x in data_dir.iterdir() if x.is_dir()]

        font_chars = {}
        for font in fonts:
            chars = [x.name for x in (self.data_dir / font).glob(f"*.{extension}")]
            font_chars[font] = chars
        return font_chars

    def load_target_font_dict(self, target_dir, font_list):
        target_font_dict = dict()
        for font in font_list:
            target_font_dict[font] = read_font(os.path.join(target_dir, font + ".ttf"))
        return target_font_dict

    def __getitem__(self, index):
        font, char = self.data_list[index]

        ref_imgs = torch.stack([self.transform(Image.open(str(self.data_dir / font / f"{rc}")))
                                for rc in self.font_ref_chars[font]])
        source_img = self.transform(render(self.source_font, char))
        target_img = self.transform(render(self.target_font_dit[font], char))

        ret = {
            "style_imgs": ref_imgs,
            "source_imgs": source_img,
            "target_imgs": target_img,
            "fonts": font,
            "chars": char,
        }

        return ret

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def collate_fn(batch):
        _ret = {}
        for dp in batch:
            for key, value in dp.items():
                saved = _ret.get(key, [])
                _ret.update({key: saved + [value]})

        ret = {
            "style_imgs": torch.stack(_ret["style_imgs"]),
            "source_imgs": torch.stack(_ret["source_imgs"]),
            "target_imgs": torch.stack(_ret["target_imgs"]),
            "fonts": _ret["fonts"],
            "chars": _ret["chars"],
        }

        return ret