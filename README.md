This project aims at simplifying playing around with LLM fine-tuning.
It uses json config files to let you specify base model, adapter and training set settings.
Comes with simple finetuning and generate scripts for testing.

Example config:
```
{
	"base" : {
		"path" : "models/13b-hf",
		"bits" : 8
	},
	"adapter" : {
		"type" : "LoRA",
		"loraModules" : "q_proj,k_proj,v_proj,o_proj",
		"loraR" : 16
	},
	"training" : {
		"dataPath" : "data/test/mydata.json",
		"epochs" : 10,
		"cutoff" : 512,
		"batchSize" : 4,
		"accumulationSteps" : 32,
		"groupByLength" : "True",
		"checkpointSteps" : 50,
		"checkpointLimit" : 50,
		"loggingSteps" : 2
	},
	"ui" : {
		"inputFields" : "instruction"
	},
	"templatePath" : "data/test/mytemplate.json"
}

```
By default output path will be same as file name of config.
All config options and their defaults can be found in modules/settings.py.
