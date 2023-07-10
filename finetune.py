#!/usr/bin/python

import fire

from modules.settings import Settings
from modules.data import DataProcessor
from modules.model import Model


def main(s : str = None):
    assert s, "Must provide settings file name"
    settings = Settings(s)
    settings.print()

    model = Model(settings)
    dp = DataProcessor(settings.templatePath)
    data = dp.loadData(settings.training.dataPath)
    print(f"Training data length: {len(data)}")
    model.train(data)


if __name__ == "__main__":
    fire.Fire(main)
