import os
import re
from typing import Iterator
from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    TextIteratorStreamer
)
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)

from .settings import Settings, TrainingSettings

# Tokenizer class exists to limit the scope of serialization in .map(). Avoids serializing entire model.
class Tokenizer:
    def __init__(self, modelPath : str, cutoff : int):
        self.tokenizer = AutoTokenizer.from_pretrained(modelPath)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 0
        self.cutoff = cutoff


    def tokenize(self, data):
        result = self.tokenizer(
            data["input"],
            truncation=True,
            max_length=self.cutoff,
            padding=False,
            return_tensors=None,
            add_special_tokens=True
        )
        result["labels"] = result["input_ids"].copy()
        return result		


    def addEOSToken(self, enable : bool):
        self.tokenizer.add_eos_token = enable


class Model:
    def __init__(self, settings : Settings, trainable=True):
        assert settings.base.bits in (4, 8, 16), '"bits" must be 4, 8 or 16'
        self.settings = settings
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.base.path,
            load_in_8bit=(settings.base.bits == 8),
            torch_dtype=torch.float16
        )
        if settings.base.bits != 16:
            self.model = prepare_model_for_kbit_training(self.model)

        adset = settings.adapter
        if adset.type != "LoRA":
            raise ValueError(f"Only LoRA supported. Requested type: {adset.type}")

        # if usable output has already been created, it wins over specified adapter
        path = self._findLoadableModel(settings.training.outputPath)
        if not path:
            path = self._findLoadableModel(adset.path)
        print(f"Loading adapter from: {path}")

        if path:
            self.model = PeftModel.from_pretrained(
                self.model,
                path,
                is_trainable=True,
                torch_dtype=torch.float16
            )
        elif trainable:
            config = LoraConfig(
                r=adset.loraR,
                lora_alpha=adset.loraAlpha,
                target_modules=adset.loraModules.split(','),
                lora_dropout=adset.loraDropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, config)

# Trainer appears to be broken for compiled model (doesnt collect proper columns from dataset)
#		self.model = torch.compile(self.model)

        self.tokenizer = Tokenizer(settings.base.path, settings.training.cutoff)


    def _modelFinalized(self, outputPath):
        modelFileName1 = os.path.join(outputPath, "adapter_model.bin")
        modelFileName2 = os.path.join(outputPath, "model.bin")
        return os.path.isfile(modelFileName1) or os.path.isfile(modelFileName2)


    def _ensureNotOverwriting(self, outputPath):
        if self._modelFinalized(outputPath):
            print("Fine-tuning has already been finished. (Found {finalModelFileName})")
            print("To redo, please remove previous result.")
            exit(1)


    def _findLatestCheckpoint(self, outputPath):
        if os.path.isdir(outputPath):
            subdirs = [os.path.join(outputPath, d) for d in os.listdir(outputPath) if os.path.isdir(os.path.join(outputPath, d))]
            if subdirs:
                return max(subdirs, key=os.path.getmtime)
        return None


    def _findLoadableModel(self, path):
        if self._modelFinalized(path):
            return path

        return self._findLatestCheckpoint(path)


    def train(self, dataSet):
        trset = self.settings.training
        self._ensureNotOverwriting(trset.outputPath)
        checkpoint = self._findLatestCheckpoint(trset.outputPath)

        self.tokenizer.addEOSToken(True)

        if checkpoint:
            print(f"Continuing fine-tune from checkpoint {checkpoint}.")
        else:
            print("Starting new fine-tune.")

        self.model.print_trainable_parameters()

        args = TrainingArguments(
            per_device_train_batch_size=trset.batchSize,
            gradient_accumulation_steps=trset.accumulationSteps,
            group_by_length=trset.groupByLength,
            warmup_steps=trset.warmupSteps,
            num_train_epochs=trset.epochs,
            learning_rate=trset.learningRate,
            weight_decay=trset.weightDecay,
            fp16=True,
            logging_steps=trset.loggingSteps,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=trset.checkpointSteps,
            output_dir=trset.outputPath,
            save_total_limit=trset.checkpointLimit
        )
        preparedDataSet = dataSet.map(self.tokenizer.tokenize)
        trainer = Trainer(
            model=self.model,
            train_dataset=preparedDataSet,
            args=args,
            data_collator=DataCollatorForSeq2Seq(
                self.tokenizer.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )
        self.model.config.use_cache = False
        trainer.train(resume_from_checkpoint=checkpoint)
        self.model.save_pretrained(trset.outputPath)


    def generate(self, input : str, limit : int = 128, temp : float = 0.1, top_p : float = 0.75, top_k : int = 40) -> Iterator[str]:
        self.model.config.use_cache = True
        self.tokenizer.addEOSToken(False)
        tokenizer = self.tokenizer.tokenizer

        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)

        config = GenerationConfig(
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            num_beams=1
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

        kwargs = dict(
            input_ids=input_ids,
            generation_config=config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=limit,
            streamer=streamer
        )

        if tokenizer.eos_token:
            eosPattern = re.compile(re.escape(tokenizer.eos_token) + '$')
        else:
            eosPattern = None

        with torch.no_grad():
            thread = Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            for text in streamer:
                if eosPattern:
                    yield eosPattern.sub('', text)
                else:
                    yield text
            thread.join()


    def embeddings(self, input):
        tokens = self.tokenizer.tokenizer(input, return_tensors='pt', add_special_tokens=False)['input_ids']
        count = tokens.shape[1]
        embeddings = self.model.model.embed_tokens(tokens)
        return embeddings.view(count, -1)

    def dumpDetails(self):
        for name, parameter in self.model.named_parameters():
            print(f"{name}  Shape: {list(parameter.shape)}")
