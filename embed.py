#!/usr/bin/python

import os

from torch.nn.functional import cosine_similarity

from modules.launcher import launch
from modules.settings import Settings
from modules.model import Model


def embed(model, t):
    emb = model.lookupEmbeddings(t)
    if emb.shape[0] != 1:
        print(f"Warning: Input {t} does not tokenize into single token.")
    return emb[0]


def diff(model, a, b):
    ea = embed(model, a)
    eb = embed(model, b)
    return cosine_similarity(ea, eb, dim=0)


def main(s : str = None, *strings):
    settings = Settings(s)
    settings.print()

    model = Model(settings, trainable=False)
    print()
    print(f"Cosine Similarity ({settings.ui.title})")
    print(f"{' ':8s}", end='')
    for x in strings:
        print(f"{x:>8s}", end='')
    print()

    for y in strings:
        print(f"{y:8s}", end='')
        for x in strings:
            print(f"{diff(model, x, y):8.3f}", end='')
        print()


if __name__ == "__main__":
    launch(main)
