"""
Integration test for model class.

Uses huggingface libs and gpt2 model from huggingface repository. (https://huggingface.co/gpt2)
Note: The model download is ~550MB. Model should be cached on disk using hf mechanisms.
"""
import os
import pytest

from torch.nn.functional import cosine_similarity

from modules.settings import Settings
from modules.data import DataProcessor
from modules.model import (Model, Tokenizer)


@pytest.fixture(scope="module", autouse=True)
def changeDirectory():
    originalCwd = os.getcwd()
    try:
        os.chdir(originalCwd + "/test/resources")
        yield
    finally:
        os.chdir(originalCwd)

@pytest.fixture(scope="module")
def inferenceModel():
    settings = Settings("settings-inference.json")
    return Model(settings)

@pytest.fixture(scope="module")
def trainingModel():
    settings = Settings("settings-training.json")
    return Model(settings)


def testUsedModel():
    for sf in ["settings-inference.json", "settings-training.json"]:
        settings = Settings(sf)
        assert settings.base.path == "gpt2", "Update documentation on top when replacing model. Also consider download size and compute times for test."

def testlookupEmbeddings(inferenceModel):
    cat = inferenceModel.lookupEmbeddings("cat")
    assert cat.shape[0] == 1
    dog = inferenceModel.lookupEmbeddings("dog")
    assert dog.shape[0] == 1
    s = cosine_similarity(cat[0], dog[0], dim=0)
    assert s > 0 and s < 1

def testfindSimilarTokens(inferenceModel):
    cat = inferenceModel.lookupEmbeddings("cat")[0]
    results = inferenceModel.findSimilarTokens(cat, 10)
    assert len(results) == 10
    assert results[0] == ("cat", 1.0)

def testGenerate(inferenceModel):
    result = inferenceModel.generate("hello", limit=5)
    for n, _ in enumerate(result, start=1):
        assert n <= 6
    assert n == 6

def testTrain(trainingModel):
    settings = Settings("settings-training.json")

    adapterFiles = ["adapter_config.json", "adapter_model.bin", "README.md"]
    adapterFiles = [ os.path.join(settings.training.outputPath, f) for f in adapterFiles ]
    deleteFiles(adapterFiles)

    dp = DataProcessor(settings.training.templatePath)
    data = dp.loadData(settings.training.dataPath)
    try:
        trainingModel.train(data)
        for f in adapterFiles:
            assert os.path.exists(f)

    except RuntimeError as re:
        assert not "unscale_() has already been called on this optimizer" in str(re),\
            f"Unscale error can be caused at epoch boundary by accelerate / transformers bug. Check versions.\n{re}"
        raise

    finally:
        deleteFiles(adapterFiles)
        os.rmdir(settings.training.outputPath)


def deleteFiles(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)
