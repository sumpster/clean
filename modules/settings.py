import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseSettings:
    path : str
    bits : int = 8


@dataclass
class AdapterSettings:
    path : str = None
    type : str = "LoRA"
    loraModules : str = None
    loraR : int = 16
    loraAlpha : int = 16
    loraDropout : float = 0.05


@dataclass
class TrainingSettings:
    dataPath : str = None
    outputPath : str = None
    templatePath : str = None

    epochs : int = 10
    learningRate : float = 3e-4
    weightDecay : float = 0
    cutoff : int = 256
    batchSize : int = 4
    accumulationSteps : int = 32
    groupByLength : bool = False
    warmupSteps : int = 100
    checkpointSteps : int = 100
    checkpointLimit : int = 5
    loggingSteps : int = 10


@dataclass
class InferenceSettings:
    templatePath : str = None

    maxLength : int = 1024
    temperature : float = 0.1
    top_p : float = 0.75
    top_k : float = 40


@dataclass
class UiSettings:
    title : str = ""
    inputFields : str = "input"


@dataclass
class Settings:
    base : BaseSettings
    adapter : AdapterSettings
    training : TrainingSettings
    inference : InferenceSettings
    ui : UiSettings
    templatePath : str = None


    def __init__(self, settingsFileName: str):
        with open(settingsFileName, 'r') as file:
            json_dict = json.load(file)

        self.base = BaseSettings(**json_dict.get('base', {}))
        self.adapter = AdapterSettings(**json_dict.get('adapter', {}))
        self.training = TrainingSettings(**json_dict.get('training', {}))
        self.inference = InferenceSettings(**json_dict.get('inference', {}))
        self.ui = UiSettings(**json_dict.get('ui', {}))

        self.templatePath = json_dict.get('templatePath', self.templatePath)
        if not self.inference.templatePath:
            self.inference.templatePath = self.templatePath
        if not self.training.templatePath:
            self.training.templatePath = self.templatePath

        defaultBase = self._getBaseName(settingsFileName)
        if not self.adapter.path:
            self.adapter.path = defaultBase
        if not self.training.outputPath:
            self.training.outputPath = defaultBase


    def _getBaseName(self, fileName : str) -> str:
        pos = fileName.rfind('.')
        if pos != -1:
            return fileName[:pos]
        else:
            return fileName


    def print(self):
        print("=== SETTINGS ========================")
        print(f"Base: {vars(self.base)}")
        print(f"Adapter: {vars(self.adapter)}")
        print(f"Training: {vars(self.training)}")
        print(f"Inference: {vars(self.inference)}")
        print(f"UI: {vars(self.ui)}")
        print(f"templatePath: {self.templatePath}")
        print("=====================================")
