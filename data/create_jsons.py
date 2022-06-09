"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""

import json

COMPONENT_RANGE = (int('3131', 16), int('3163', 16))  # kr 자음/모음
COMPLETE_RANGE = (int('ac00', 16), int('d7a3', 16))   # kr all complete chars
COMPLETE_SET = frozenset(chr(code) for code in range(COMPLETE_RANGE[0], COMPLETE_RANGE[1]+1))
COMPLETE_LIST = sorted(COMPLETE_SET)

CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
            'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
             'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
             'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
             'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

N_CHO, N_JUNG, N_JONG = len(CHO_LIST), len(JUNG_LIST), len(JONG_LIST)
N_COMPONENTS = N_CHO + N_JUNG + N_JONG


def compose(cho, jung, jong):
    """Compose ids to char"""
    char_id = cho * N_JONG * N_JUNG + jung * N_JONG + jong + COMPLETE_RANGE[0]
    return chr(char_id)


def decompose(char):
    """Decompose char to ids"""
    char_code = ord(char)
    if COMPLETE_RANGE[0] <= char_code <= COMPLETE_RANGE[1]:
        char_code -= COMPLETE_RANGE[0]
        jong = char_code % N_JONG
        jung = (char_code // N_JONG) % N_JUNG
        cho = char_code // (N_JONG * N_JUNG)
        char_id = (cho, jung, jong)
    elif COMPONENT_RANGE[0] <= char_code <= COMPONENT_RANGE[1]:
        char_code -= COMPONENT_RANGE[0]
        # raise ValueError('Component only ({})'.format(char))
        return None
    else:
        # raise ValueError('{} is Non kor'.format(char))
        return None

    return char_id


def get_all_chars(txt_dir):
    with open(txt_dir) as f:
        all_chars = f.readlines()[0]
    return all_chars


def get_all_chars(txt_dir, gen=False):
    with open(txt_dir) as f:
        if not gen:
            all_chars = f.readlines()[0]
        else:
            all_chars = f.read().splitlines()
    return all_chars


def save_decomposition_json(txt_dir, out_dir):
    decomposition_dict = dict()
    all_chars = get_all_chars(txt_dir)

    for char in all_chars:
        char_decomposition = decompose(char)
        if char_decomposition == None:
            continue

        cho, jung, jong = CHO_LIST[char_decomposition[0]], JUNG_LIST[char_decomposition[1]], JONG_LIST[char_decomposition[2]]
        if jong == ' ': # doesn't have jongsung
            decomposition_dict[char] = [cho, jung]
        else:
            decomposition_dict[char] = [cho, jung, jong]

    with open(out_dir, "w") as f:
        f.write(json.dumps(decomposition_dict))


def save_primal_json(out_dir):
    primal_list = list(set(CHO_LIST + JUNG_LIST + JONG_LIST))
    primal_list.remove(" ")
    with open(out_dir, "w") as f:
        f.write(json.dumps(primal_list))


def save_gen_json(txt_dir, out_dir):
    gen_list = get_all_chars(txt_dir, gen=True)

    with open(out_dir, "w") as f:
        f.write(json.dumps(gen_list))


if __name__ == "__main__":
    txt_dir = "/home/server17/changhun_workspace/AI604/mxfont/data/ttfs/train/나눔손글씨 중학생.txt"
    decomposition_out_dir = "/home/server17/changhun_workspace/AI604/mxfont/data/korean_decomposition.json"
    primals_out_dir = "/home/server17/changhun_workspace/AI604/mxfont/data/korean_primals.json"
    gen_txt_dir = "/home/server08/changhun_workspace/FFG-Korean/data/val_content.txt"
    gen_out_dir = "korean_gen.json"

    save_decomposition_json(txt_dir, decomposition_out_dir)
    save_primal_json(primals_out_dir)
    save_gen_json(gen_txt_dir, gen_out_dir)