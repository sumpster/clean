#!/usr/bin/python

import os

from modules.launcher import launch
from modules.settings import Settings
from modules.model import Model


def main(s : str = None):
	settings = Settings(s)
	settings.print()

	model = Model(settings, trainable=False)
	model.dumpDetails()


if __name__ == "__main__":
    launch(main)
