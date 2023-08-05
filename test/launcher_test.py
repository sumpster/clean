import sys
import os
import pytest

from modules.launcher import launch


def testSettingsOnly():
    sys.argv = ["test.py", "test/resources/settings.json"]
    called = False
    def func(settingsPath, second=None):
        nonlocal called
        called = True
        assert settingsPath == "settings.json"
        assert second == None

    launch(func)
    assert called, "Function was never called."

def testMultiArg():
    sys.argv = ["test.py", "test/resources/settings.json", "more"]
    called = False
    def func(settingsPath, second=None):
        nonlocal called
        called = True
        assert settingsPath == "settings.json"
        assert second == "more"

    launch(func)
    assert called, "Function was never called."

def testDirectoryChange():
    sys.argv = ["test.py", "test/resources/settings.json"]
    original = os.getcwd()
    called = False
    def func(settingsPath, second=None):
        nonlocal called
        called = True
        assert os.getcwd()[len(original):] == "/test/resources"

    launch(func)
    assert os.getcwd() == original
    assert called, "Function was never called."

def testDirectorySwitchBackOnException():
    sys.argv = ["test.py", "test/resources/settings.json"]
    original = os.getcwd()
    called = False
    def func(settingsPath, second=None):
        nonlocal called
        called = True
        raise RuntimeError("exception")

    try:
        launch(func)
    except RuntimeError:
        pass

    assert os.getcwd() == original
    assert called, "Function was never called."
