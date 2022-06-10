import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image, ImageFont, ImageDraw
import numpy as np


def read_font(fontfile, size=150):
    font = ImageFont.truetype(str(fontfile), size=size)
    return font


def render(font, char, size=(128, 128), pad=20):
    width, height = font.getsize(char)
    max_size = max(width, height)

    if width < height:
        start_w = (height - width) // 2 + pad
        start_h = pad
    else:
        start_w = pad
        start_h = (width - height) // 2 + pad

    img = Image.new("L", (max_size+(pad*2), max_size+(pad*2)), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_w, start_h), char, font=font)
    img = img.resize(size, 2)
    return img


out_folder = "example_images"
base_folders = [
    "/home/server08/changhun_workspace/FFG-Korean/eval_baseline",
    "/home/server08/changhun_workspace/FFG-Korean/eval_postnet",
    "/home/server08/changhun_workspace/FFG-Korean/eval_patchgan",
    "/home/server08/changhun_workspace/FFG-Korean/eval_consistent",
    "/home/server08/changhun_workspace/FFG-Korean/eval_high_pixel",
    "/home/server08/changhun_workspace/FFG-Korean/eval_very_high_pixel",
    "/home/server08/changhun_workspace/FFG-Korean/eval_consistent_nopixel",
    "/home/server08/changhun_workspace/FFG-Korean/eval_no_ground_truth_2",
]
original_font = "/home/server08/changhun_workspace/FFG-Korean/data/ttfs/val/나눔손글씨 또박또박.ttf"
fonts = [
    "나눔손글씨 고딕 아니고 고딩",
    "나눔손글씨 김유이체",
    "나눔손글씨 꽃내음",
    "나눔손글씨 부장님 눈치체",
    "나눔손글씨 아빠의 연애편지",
    "나눔손글씨 암스테르담",
    "나눔손글씨 예쁜 민경체",
    "나눔손글씨 옥비체",
    "나눔손글씨 유니 띵땅띵땅",
    "나눔손글씨 코코체"
]

with open("/home/server08/changhun_workspace/FFG-Korean/data/korean_gen.json", "r") as f:
    chars = json.load(f)

selected_chars = random.sample(chars, 10)
random.shuffle(selected_chars)
random.shuffle(fonts)

f, axarr = plt.subplots(10, 10)

[axi.set_axis_off() for axi in axarr.ravel()]

for j, char in enumerate(selected_chars):
    axarr[0, j].imshow(render(read_font(original_font), str(char[0])), cmap=plt.get_cmap('gray'))

for j, (font, char) in enumerate(zip(fonts, selected_chars)):
    print(font, char)
    font_path = os.path.join(*[
        "/home/server08/changhun_workspace/FFG-Korean/data/ttfs/val",
        font + ".ttf"
    ])

    axarr[1, j].imshow(render(read_font(font_path), str(char[0])), cmap=plt.get_cmap('gray'))

for i, base_folder in enumerate(base_folders):
    for j, (font, char) in enumerate(zip(fonts, selected_chars)):
        image_path = os.path.join(*[
            base_folder,
            font,
            char + ".png"
        ])
        image = mpimg.imread(image_path)

        axarr[i + 2, j].imshow(image, cmap=plt.get_cmap('gray'))

plt.savefig("example_images.png")