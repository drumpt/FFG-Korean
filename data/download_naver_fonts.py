import os
import math
import requests

from sklearn.model_selection import train_test_split
    

def download(url, out_path="."):
    r = requests.get(url)

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(out_path, 'wb') as f:
        f.write(r.content)


def get_font_list(font_path):
    with open(font_path) as f:
        font_list = f.read().splitlines()
    return font_list


if __name__ == "__main__":
    font_path = "naver_font_list.txt"
    train_path, valid_path = "ttfs/train", "ttfs/val"
    base_url = "https://ssl.pstatic.net/static/clova/service/clova_ai/event/handwriting/download/"
    
    font_list = get_font_list(font_path=font_path)
    train_font_list, valid_font_list = train_test_split(font_list, test_size=0.1, random_state=42)

    print(train_font_list, len(train_font_list))
    print(valid_font_list, len(valid_font_list))

    for train_font in train_font_list:
        url = base_url + train_font.replace(" ", "%20") + ".ttf"
        download(url, os.path.join(train_path, train_font + ".ttf"))
    for valid_font in valid_font_list:
        url = base_url + valid_font.replace(" ", "%20") + ".ttf"
        download(url, os.path.join(valid_path, valid_font + ".ttf"))