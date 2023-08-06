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

def testEmbeddings(inferenceModel):
    cat = inferenceModel.embeddings("cat")
    assert cat.shape[0] == 1
    dog = inferenceModel.embeddings("dog")
    assert dog.shape[0] == 1
    s = cosine_similarity(cat[0], dog[0], dim=0)
    assert s > 0 and s < 1

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
    finally:
        deleteFiles(adapterFiles)
        os.rmdir(settings.training.outputPath)

def deleteFiles(files):
    for f in files:
        if os.path.exists(f):
            os.remove(f)
