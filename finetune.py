#!/usr/bin/python
import os

from modules.launcher import launch
from modules.settings import Settings
from modules.data import DataProcessor
from modules.model import Model


def main(s : str = None):
	settings = Settings(s)
	settings.print()

	model = Model(settings)
	dp = DataProcessor(settings.templatePath)
	data = dp.loadData(settings.training.dataPath)
	print(f"Training data length: {len(data)}")
	model.train(data)


if __name__ == "__main__":
    launch(main)
