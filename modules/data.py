import re
import json

import datasets

from .template import Template

# Note: If not template is used, input field of data is used for training.
class DataProcessor:
	def __init__(self, templatePath = None):
		self.template = Template(templatePath)


	def _applyTemplate(self, data):
		return { "input" : self.template.apply(**data) }


	def loadData(self, dataPath, randomize=True):
		if dataPath.endswith(".json"):
			dataset = datasets.load_dataset("json", data_files=dataPath)
		else:
			dataset = datasets.load_dataset(dataPath)
		
		data = dataset["train"]
		if randomize:
			data = data.shuffle()
		if self.template.hasTemplate():
			data = data.map(self._applyTemplate)
		return data
			
