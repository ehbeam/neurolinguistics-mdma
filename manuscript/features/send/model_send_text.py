import os
import torch
import numpy as np
import pandas as pd
from ordered_set import OrderedSet
from datasets import Dataset
from utils import rmse_loss, compute_metrics, run_finetuning 
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification


def preprocess_text_data(raw_text_path, raw_label_path, segment_duration, splits=["train", "valid", "test"]):

    print("-"*100 + f"\n{segment_duration}-SECOND CLIPS\n" + "-"*100)
    
    data = {}
    
    for split in splits:
        
        data[split] = {"id": [], "text": [], "label": []}
        text_lengths = []
        
        split_path = f"{raw_text_path}/{split}"
        text_files = sorted(os.listdir(split_path))
        
        for text_file in text_files: 

            segment_i = 0
            video_i = text_file.split("_text")[0]
            
            text_df = pd.read_csv(f"{split_path}/{text_file}", on_bad_lines="skip")
            text_df["words"] = text_df["words"].astype(str)
            text_df = pd.DataFrame(np.repeat(text_df.values, 10, axis=0), columns=text_df.columns)
            
            rating_i = "".join([s for s in video_i if s.isdigit() or s == "_"])
            rating_df = pd.read_csv(f"{raw_label_path}/{split}/observer_EWE/results_{rating_i}.csv") # Ratings are q0.5s
    
            df = pd.DataFrame({"time": rating_df["time"], 
                               "words": text_df["words"], 
                               "evaluatorWeightedEstimate": rating_df["evaluatorWeightedEstimate"]})
            df = df.dropna()
            df = df.groupby(np.arange(len(df)) // (segment_duration/0.5)).agg({"time": "min", 
                                                                               "words": lambda x: " ".join(OrderedSet(x)),
                                                                               "evaluatorWeightedEstimate": "mean"})
            
            for words, label in zip(df["words"], df["evaluatorWeightedEstimate"]):
                n_words = len(str(words).split())
                if n_words > 1:
                    segment_i += 1
                    data[split]["id"] += [f"{video_i}_{segment_i}"]
                    data[split]["text"] += [str(words).strip()]
                    data[split]["label"] += [label/100.0]
    
                text_lengths += [n_words]

        df = pd.DataFrame({"id": data[split]["id"], "text": data[split]["text"], "label": data[split]["label"]})
        df.to_csv(f"data/texts/texts_{segment_duration}s_{split}.csv", index=None)
        
        print(f"{split.upper()} SPLIT")
        print(f"Number of texts:     {len(text_lengths)}")
        print(f"Max words/text:      {max(text_lengths)}")
        print(f"Median words/text:   {np.median(text_lengths):g}")
        print(f"Mean words/text:     {np.mean(text_lengths):4.2f}")
        print(f"Std dev words/text:  {np.std(text_lengths):4.2f}\n")
    
    return data

    
def load_text_inputs(text_path, segment_duration, tokenizer, max_length=100, splits=["train", "valid", "test"]):
    
    print("-"*100 + f"\n{segment_duration}-SECOND CLIPS\n" + "-"*100)

    data = {}
    for split in splits:
        df = pd.read_csv(f"{text_path}/texts_{segment_duration}s_{split}.csv", index_col="id")
        data[split] = {"text": df["text"], "label": df["label"]}
    
    def tokenize(examples):
        return tokenizer(examples["text"], max_length=max_length, padding="max_length", truncation=True) 

    print("\nTOKENIZING TEXTS")
    inputs = {}
    for split in splits:
        inputs[split] = Dataset.from_dict(data[split]).map(tokenize, batched=True)
    print("\n" + "-"*100 + "\n")

    return inputs


###################################################################################################
############################################# ROBERTA #############################################
###################################################################################################

def load_roberta_path(model_name, inputs, **kwargs):

    if "model_nickname" in kwargs.keys():
        model_name = kwargs["model_nickname"]

    model_path = f"models/{kwargs["segment_duration"]}s/{model_name}/{model_name}_{kwargs["segment_duration"]:g}s_epochs{kwargs["n_epochs"]}_lr{kwargs["lr"]}_drop-hidden{kwargs["hidden_dropout"]:g}_drop-attn{kwargs["attn_dropout"]:g}_drop-clf{kwargs["clf_dropout"]:g}_tune{kwargs["tune_layer"]}"
    
    return model_path

    
def load_roberta_trainer(model_name, tokenizer, inputs, **kwargs):

    model_path = load_roberta_path(model_name, inputs, **kwargs)
            
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=kwargs["batch_size"]):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = rmse_loss(labels, logits)
            return (loss, outputs) if return_outputs else loss
            
    training_args = TrainingArguments(output_dir=model_path, 
                                      learning_rate=kwargs["lr"],
                                      per_device_train_batch_size=kwargs["batch_size"],
                                      per_device_eval_batch_size=kwargs["batch_size"],
                                      eval_strategy="epoch", 
                                      logging_strategy="epoch",
                                      save_strategy="no",
                                      load_best_model_at_end=False,
                                      num_train_epochs=kwargs["n_epochs"],
                                      warmup_steps=kwargs["warmup_steps"],
                                      save_total_limit=1)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               num_labels=1, 
                                                               ignore_mismatched_sizes=True,
                                                               hidden_dropout_prob=kwargs["hidden_dropout"],
                                                               attention_probs_dropout_prob=kwargs["attn_dropout"],
                                                               classifier_dropout=kwargs["clf_dropout"]).to(kwargs["device"])

    if os.path.exists(f"{model_path}/model.safetensors"):
        print(f"\nFOUND MODEL\nPath: {model_path}\n")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(kwargs["device"])

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "roberta" in name and f".{str(kwargs["tune_layer"])}." in name: 
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True

    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=inputs["train"],
                            eval_dataset=inputs["valid"],
                            processing_class=tokenizer,
                            compute_metrics=compute_metrics)

    return trainer


def run_roberta(model_name, tokenizer, inputs, **kwargs):
    
    model_path = load_roberta_path(model_name, inputs, **kwargs)
    trainer = load_roberta_trainer(model_name, tokenizer, inputs, **kwargs)

    param_names = ["segment_duration", "n_epochs", "lr", "hidden_dropout", "attn_dropout", "clf_dropout", "tune_layer"]
    param_dict = {param_name: kwargs[param_name] for param_name in param_names}
    
    model_path, evaluation = run_finetuning(model_path, trainer, param_dict, inputs, plot=kwargs["plot"], splits=kwargs["splits"])
    
    return model_path, evaluation


###################################################################################################
############################################## LLaMA ##############################################
###################################################################################################

def load_llama_path(model_name, inputs, **kwargs):

    model_path = f"models/{kwargs["segment_duration"]}s/{model_name}/{model_name}_{kwargs["segment_duration"]:g}s_epochs{kwargs["n_epochs"]}_lr{kwargs["lr"]}_drop-attn{kwargs["attn_dropout"]:g}_tune{kwargs["tune_layer"]}"
    
    return model_path

    
def load_llama_trainer(model_name, tokenizer, inputs, **kwargs):

    model_path = load_llama_path(model_name, inputs, **kwargs)
            
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=kwargs["batch_size"]):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = rmse_loss(labels, logits)
            return (loss, outputs) if return_outputs else loss
            
    training_args = TrainingArguments(output_dir=model_path, 
                                      learning_rate=kwargs["lr"],
                                      per_device_train_batch_size=kwargs["batch_size"],
                                      per_device_eval_batch_size=kwargs["batch_size"],
                                      eval_strategy="epoch", 
                                      logging_strategy="epoch",
                                      save_strategy="no",
                                      load_best_model_at_end=False,
                                      num_train_epochs=kwargs["n_epochs"],
                                      warmup_steps=kwargs["warmup_steps"],
                                      save_total_limit=1)
    
    model = AutoModelForSequenceClassification.from_pretrained(f"meta-llama/{model_name}", 
                                                               num_labels=1, 
                                                               ignore_mismatched_sizes=True,
                                                               attention_dropout=kwargs["attn_dropout"],
                                                               token=kwargs["access_token"]).to(kwargs["device"])

    if os.path.exists(f"{model_path}/model.safetensors"):
        print(f"\nFOUND MODEL\nPath: {model_path}\n")
        model = AutoModelForSequenceClassification.from_pretrained(model_path, token=kwargs["access_token"]).to(kwargs["device"])

    for name, param in model.named_parameters():
        param.requires_grad = False
        if f"model.layers.{str(kwargs["tune_layer"])}." in name: 
            param.requires_grad = True
        if "model.norm.weight" in name:
            param.requires_grad = True
        if "score.weight" in name:
            param.requires_grad = True

    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=inputs["train"],
                            eval_dataset=inputs["valid"],
                            processing_class=tokenizer,
                            compute_metrics=compute_metrics)

    return trainer


def run_llama(model_name, tokenizer, inputs, **kwargs):
    
    model_path = load_llama_path(model_name, inputs, **kwargs)
    trainer = load_llama_trainer(model_name, tokenizer, inputs, **kwargs)

    param_names = ["segment_duration", "n_epochs", "lr", "attn_dropout", "tune_layer"]
    param_dict = {param_name: kwargs[param_name] for param_name in param_names}
    
    model_path, evaluation = run_finetuning(model_path, trainer, param_dict, inputs, plot=kwargs["plot"], splits=kwargs["splits"])
    
    return model_path, evaluation
    

###################################################################################################
############################################## GPT2 ###############################################
###################################################################################################

def load_gpt2_path(model_name, inputs, **kwargs):

    model_path = f"models/{kwargs["segment_duration"]}s/{model_name}/{model_name}_{kwargs["segment_duration"]:g}s_epochs{kwargs["n_epochs"]}_lr{kwargs["lr"]}_drop-resid{kwargs["resid_dropout"]:g}_drop-embed{kwargs["embed_dropout"]:g}_drop-attn{kwargs["attn_dropout"]:g}_tune{kwargs["tune_layer"]}"
    
    return model_path
    

def load_gpt2_trainer(model_name, tokenizer, inputs, **kwargs):

    model_path = load_gpt2_path(model_name, inputs, **kwargs)
            
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=kwargs["batch_size"]):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = rmse_loss(labels, logits)
            return (loss, outputs) if return_outputs else loss
            
    training_args = TrainingArguments(output_dir=model_path, 
                                      learning_rate=kwargs["lr"],
                                      per_device_train_batch_size=kwargs["batch_size"],
                                      per_device_eval_batch_size=kwargs["batch_size"],
                                      eval_strategy="epoch", 
                                      logging_strategy="epoch",
                                      save_strategy="no",
                                      load_best_model_at_end=False,
                                      num_train_epochs=kwargs["n_epochs"],
                                      warmup_steps=kwargs["warmup_steps"],
                                      save_total_limit=1)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               num_labels=1, 
                                                               ignore_mismatched_sizes=True,
                                                               resid_pdrop=kwargs["resid_dropout"],
                                                               embd_pdrop=kwargs["embed_dropout"],
                                                               attn_pdrop=kwargs["attn_dropout"]).to(kwargs["device"])

    if os.path.exists(f"{model_path}/model.safetensors"):
        print(f"\nFOUND MODEL\nPath: {model_path}\n")
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(kwargs["device"])

    for name, param in model.named_parameters():
        param.requires_grad = False
        if str(kwargs["tune_layer"]) in name: 
            param.requires_grad = True
        if "transformer.ln" in name: 
            param.requires_grad = True
        if "score.weight" in name: 
            param.requires_grad = True

    model.config.pad_token_id = model.config.eos_token_id
    
    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=inputs["train"],
                            eval_dataset=inputs["valid"],
                            processing_class=tokenizer,
                            compute_metrics=compute_metrics)

    return trainer


def run_gpt2(model_name, tokenizer, inputs, **kwargs):
    
    model_path = load_gpt2_path(model_name, inputs, **kwargs)
    trainer = load_gpt2_trainer(model_name, tokenizer, inputs, **kwargs)

    param_names = ["segment_duration", "n_epochs", "lr", "resid_dropout", "embed_dropout", "attn_dropout", "tune_layer"]
    param_dict = {param_name: kwargs[param_name] for param_name in param_names}
    
    model_path, evaluation = run_finetuning(model_path, trainer, param_dict, inputs, plot=kwargs["plot"], splits=kwargs["splits"])
    
    return model_path, evaluation

