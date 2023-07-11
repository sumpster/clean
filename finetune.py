#!/usr/bin/python
import os

import fire

from modules.settings import Settings
from modules.data import DataProcessor
from modules.model import Model


def main(s : str = None):
    originalCwd = os.getcwd()
    try:
        assert s, "Must provide settings file name"
        baseDir = os.path.dirname(s)
        os.chdir(baseDir)

        settings = Settings(os.path.basename(s))
        settings.print()

        model = Model(settings)
        dp = DataProcessor(settings.templatePath)
        data = dp.loadData(settings.training.dataPath)
        print(f"Training data length: {len(data)}")
        model.train(data)

    finally:
        os.chdir(originalCwd)


if __name__ == "__main__":
    fire.Fire(main)
