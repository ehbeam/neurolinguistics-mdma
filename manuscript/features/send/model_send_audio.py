import os

import pydub
import librosa
import soundfile as sf
from moviepy.editor import VideoFileClip

from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import WhisperFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import WhisperForAudioClassification, Wav2Vec2ForSequenceClassification

import numpy as np
import pandas as pd

from utils import rmse_loss, compute_metrics, run_finetuning


def preprocess_audio_data(raw_video_path, raw_label_path, segment_duration, padded_duration=30, sampling_rate=16000, splits=["train", "valid", "test"], verbose=True):

    if verbose:
        print("-"*100 + f"\n{segment_duration}-SECOND CLIPS\n" + "-"*100)

    full_out_path = "data/audio/full"
    if not os.path.exists(full_out_path):
        os.mkdir(full_out_path)
        
    segment_out_path = f"data/audio/{segment_duration:g}s"
    if not os.path.exists(segment_out_path):
        os.mkdir(segment_out_path)

    data = {}
    for split in splits:

        if not os.path.exists(f"{full_out_path}/{split}"):
            os.mkdir(f"{full_out_path}/{split}")
        
        if not os.path.exists(f"{segment_out_path}/{split}"):
            os.mkdir(f"{segment_out_path}/{split}")
        
        data[split] = {"id": [], "audio": [], "path": [], "label": []}

        split_video_files = sorted(os.listdir(f"{raw_video_path}/{split}"))
        for video_file in split_video_files:
            
            video_i = video_file[:10]
            full_out_file = f"{full_out_path}/{split}/{video_i}.wav"

            # Load labels for the video
            rating_i = "".join([s for s in video_i if s.isdigit() or s == "_"])
            rating_df = pd.read_csv(f"{raw_label_path}/{split}/observer_EWE/results_{rating_i}.csv")
            
            # Extract audio from video if needed, otherwise load full-length audio
            if not os.path.exists(full_out_file):
                audio = VideoFileClip(f"{raw_video_path}/{split}/{video_file}").audio
                audio.write_audiofile(full_out_file, codec="pcm_s16le", verbose=False)

            # Resample to target sampling_rate
            audio, orig_sr = librosa.load(full_out_file, sr=None)
            audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=sampling_rate)
            sf.write(full_out_file, audio, sampling_rate)            
            audio = pydub.AudioSegment.from_file(full_out_file)

            n_segments = int(len(audio) / (segment_duration * 1000)) + 1
            for segment_i in range(n_segments):
                
                out_file = f"{segment_out_path}/{split}/{video_i}_seg{segment_i:03d}.wav"

                # Truncate to segment_duration
                start_segment = segment_duration * 1000 * segment_i
                end_segment = min(start_segment + (segment_duration * 1000), len(audio))
                audio_segment = audio[start_segment:end_segment]
                audio_segment.export(out_file, format="wav")
                
                # Pad to padded_duration
                audio_segment, sampling_rate = librosa.load(out_file, sr=sampling_rate)
                if len(audio_segment) < int(sampling_rate * padded_duration):
                    audio_segment = np.pad(audio_segment, (0, int(sampling_rate * padded_duration) - len(audio_segment)))
                sf.write(out_file, audio_segment, sampling_rate)

                # Average labels within segment_duration
                rating_bin_df = rating_df.loc[(rating_df["time"] >= (start_segment/1000)) & (rating_df["time"] < (end_segment/1000))]
                rating_bin_mean = float(rating_bin_df["evaluatorWeightedEstimate"].mean()) / 100.0
                                
                data[split]["id"] += [f"{video_i}_{segment_i}"]
                data[split]["audio"] += [audio_segment]
                data[split]["path"] += [out_file]
                data[split]["label"] += [rating_bin_mean]
        
        df = pd.DataFrame({"id": data[split]["id"], "path": data[split]["path"], "label": data[split]["label"]})
        df.to_csv(f"data/audio/audio_{segment_duration}s_{split}.csv", index=None)

        if verbose:
            print(f"{split.upper()} SPLIT")
            print(f"Number of clips: {len(data[split]['path'])}\n")
        
    return data
    
    
def load_audio_inputs(data, segment_duration, feature_extractor, 
                      splits=["train", "validation", "test"]):

    print("-"*100 + f"\n{segment_duration}-SECOND CLIPS\n" + "-"*100)
    
    def process(examples):
        return feature_extractor(examples["audio"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")

    inputs = {}
    for split in splits:
        inputs[split] = Dataset.from_dict(data[split]).map(process, batched=True)
    print("\n" + "-"*100 + "\n")
    
    return inputs
    

###################################################################################################
############################################# WAV2VEC #############################################
###################################################################################################

def load_wav2vec2_path(model_name, inputs, **kwargs):

    model_path = f"models/{kwargs['segment_duration']}s/{model_name}/{model_name}_{kwargs['segment_duration']}s_epochs{kwargs['n_epochs']}_lr{kwargs['lr']}_drop-hidden{kwargs['hidden_dropout']:g}_drop-actv{kwargs['actv_dropout']:g}_drop-attn{kwargs['attn_dropout']:g}_drop-final{kwargs['final_dropout']:g}_tune{kwargs['tune_layer']}"
    
    return model_path


def load_wav2vec2_trainer(model_name, inputs, **kwargs):

    model_path = load_wav2vec2_path(model_name, inputs, **kwargs)
            
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
                                      num_train_epochs=kwargs["n_epochs"],
                                      warmup_steps=kwargs["warmup_steps"],
                                      save_total_limit=1)
    
    model = Wav2Vec2ForSequenceClassification.from_pretrained(f"facebook/{model_name}",
                                                              num_labels=1,
                                                              ignore_mismatched_sizes=True,
                                                              hidden_dropout=kwargs["hidden_dropout"],
                                                              activation_dropout=kwargs["actv_dropout"],
                                                              attention_dropout=kwargs["attn_dropout"],
                                                              final_dropout=kwargs["final_dropout"]).to(kwargs["device"])      

    if os.path.exists(f"{model_path}/model.safetensors"):
        print(f"\nFOUND MODEL\nPath: {model_path}\n")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path).to(kwargs["device"])

    for name, param in model.named_parameters():
        param.requires_grad = False
        if f"wav2vec2.encoder.layers.{kwargs['tune_layer']}." in name:
            param.requires_grad = True
        if name.startswith("projector."):
            param.requires_grad = True
        if name.startswith("classifier."):
            param.requires_grad = True
    #     print(f"{name:70s} {param.requires_grad}")
    # print()
    
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/{model_name}")
    
    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=inputs["train"],
                            eval_dataset=inputs["valid"],
                            processing_class=feature_extractor,
                            compute_metrics=compute_metrics)

    return trainer


def run_wav2vec2(model_name, inputs, **kwargs):
    
    model_path = load_wav2vec2_path(model_name, inputs, **kwargs)
    trainer = load_wav2vec2_trainer(model_name, inputs, **kwargs)

    param_names = ["segment_duration", "n_epochs", "lr", "hidden_dropout", "actv_dropout", "attn_dropout", "final_dropout", "tune_layer"]
    param_dict = {param_name: kwargs[param_name] for param_name in param_names}
    
    model_path, evaluation = run_finetuning(model_path, trainer, param_dict, inputs, plot=kwargs["plot"], splits=kwargs["splits"])
    
    return model_path, evaluation
    

###################################################################################################
############################################# WHISPER #############################################
###################################################################################################

def load_whisper_path(model_name, inputs, **kwargs):

    model_path = f"models/{kwargs['segment_duration']}s/{model_name}/{model_name}_{kwargs['segment_duration']}s_epochs{kwargs['n_epochs']}_lr{kwargs['lr']}_drop-conn{kwargs['conn_dropout']:g}_drop-actv{kwargs['actv_dropout']:g}_drop-attn{kwargs['attn_dropout']:g}_tune{kwargs['tune_layer']}"
    
    return model_path


def load_whisper_trainer(model_name, inputs, **kwargs):

    model_path = load_whisper_path(model_name, inputs, **kwargs)
            
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
                                      num_train_epochs=kwargs["n_epochs"],
                                      warmup_steps=kwargs["warmup_steps"],
                                      save_total_limit=1)
    
    model = WhisperForAudioClassification.from_pretrained(f"openai/{model_name}",
                                                          num_labels=1,
                                                          ignore_mismatched_sizes=True,
                                                          dropout=kwargs["conn_dropout"],
                                                          attention_dropout=kwargs["attn_dropout"],
                                                          activation_dropout=kwargs["actv_dropout"]).to(kwargs["device"])      

    if os.path.exists(f"{model_path}/model.safetensors"):
        print(f"\nFOUND MODEL\nPath: {model_path}\n")
        model = WhisperForAudioClassification.from_pretrained(model_path).to(kwargs["device"])

    for name, param in model.named_parameters():
        param.requires_grad = False
        if f".{kwargs['tune_layer']}." in name:
            param.requires_grad = True
        if ".layer_norm." in name:
            param.requires_grad = True
        if "projector." in name:
            param.requires_grad = True
        if "classifier." in name:
            param.requires_grad = True
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{model_name}")
    
    trainer = CustomTrainer(model=model,
                args=training_args,
                train_dataset=inputs["train"],
                eval_dataset=inputs["valid"],
                processing_class=feature_extractor,
                compute_metrics=compute_metrics)

    return trainer


def run_whisper(model_name, inputs, **kwargs):
    
    model_path = load_whisper_path(model_name, inputs, **kwargs)
    trainer = load_whisper_trainer(model_name, inputs, **kwargs)

    param_names = ["segment_duration", "n_epochs", "lr", "conn_dropout", "attn_dropout", "actv_dropout", "tune_layer"]
    param_dict = {param_name: kwargs[param_name] for param_name in param_names}
    
    model_path, evaluation = run_finetuning(model_path, trainer, param_dict, inputs, plot=kwargs["plot"], splits=kwargs["splits"])
    
    return model_path, evaluation


