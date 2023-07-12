import os
import sys

from typing import Callable

def launch(func : Callable):
    scriptName = sys.argv[0]
    if len(sys.argv) == 1:
        print(f"Usage: {scriptName} <settingFile> ...")
        exit(1)

    settingsPath = sys.argv[1]
    remainder = sys.argv[2:]

    originalCwd = os.getcwd()
    try:
        baseDir = os.path.dirname(settingsPath)
        fileName = os.path.basename(settingsPath)
        if baseDir != "":
            os.chdir(baseDir)

        func(fileName, *remainder)

    finally:
        os.chdir(originalCwd)
