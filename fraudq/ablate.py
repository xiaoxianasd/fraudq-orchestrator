"""ablate — 消融实验开关（仅实验用，默认全部关闭）。"""

CFG = {"fixed_window": False, "all_branches": False, "no_replay": False}


def reset():
    for k in CFG:
        CFG[k] = False
