import utils

import os
import requests
import json
import re
import textwrap
import pandas as pd
import numpy as np

import nltk
import spacy
import parselmouth
from parselmouth.praat import call
from pydub.silence import detect_silence
from pydub import AudioSegment

from datasets import Dataset
from transformers import RobertaTokenizer, AutoModelForSequenceClassification
from transformers import WhisperFeatureExtractor, WhisperForAudioClassification
from transformers import Trainer, TrainingArguments

from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


########################################################################
############################## DATA SPLITS #############################
########################################################################

def split_participants(speech_df, seed=9, splits=["train", "valid", "test"]):
    
    participants = sorted(list(set(speech_df["participant"])))
    participant2biotype = {participant.split("_")[0]: biotype for participant, biotype in zip(speech_df.index, speech_df["biotype"])}

    splits_file = "data/participant_splits.csv"
    split2participants = {}
    splits_df = pd.DataFrame()
    if not os.path.exists(splits_file):
        for biotype in biotypes:
            biotype_participants = [p for p in participants if participant2biotype[p] == biotype]
            participant_ids = [int(p.replace("P", "")) for p in biotype_participants]
            split_train, split_temp = train_test_split(
                np.array(participant_ids), test_size=0.4, random_state=seed)
            split_valid, split_test = train_test_split(
                np.array(split_temp), test_size=0.5, random_state=seed)
            split2participants = {
                split: [f"P{int(p):03d}" for p in p_ids] for split, p_ids in zip(splits, [split_train, split_valid, split_test])}
            participants_list, splits_list = [], []
            for split in splits:
                for participant in split2participants[split]:
                    participants_list += [participant]
                    splits_list += [split]
            biotype_splits_df = pd.DataFrame({"participant": participants_list, "split": splits_list})
            splits_df = pd.concat([splits_df, biotype_splits_df])
        splits_df["biotype"] = [participant2biotype[line_id] for line_id in splits_df["participant"]]
        splits_df.to_csv(splits_file, index=None)
    
    else:
        splits_df = pd.read_csv(splits_file, index_col=None)
    
    for split in ["train", "valid", "test"]:
        split2participants[split] = list(splits_df[splits_df["split"] == split]["participant"])

    participant2split = {}
    for split in splits:
        for participant in split2participants[split]:
            participant2split[participant] = split
            
    return participant2split


def load_participant_splits(df, participant2split, seed=9, splits=["train", "valid", "test"]):
    split_list = []
    for participant in df["participant"]:
        # if isinstance(participant, int):
        #     participant = f"P{participant:03d}"
        if participant in participant2split.keys():
            split_list += [participant2split[participant]]
        else:
            split_list += [np.nan]
    df["split"] = split_list
    return df


########################################################################
######################## CANDIDATE NLP FEATURES ########################
########################################################################

def rmse_loss(y_true, y_pred):
    y_true = torch.Tensor(y_true)
    y_pred = torch.Tensor(y_pred)
    return torch.sqrt(torch.mean((y_true-y_pred)**2))


#################### Text-Derived Affective Valence ####################

def load_text_inputs(text_list, tokenizer, max_length=100):

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False) # max_length=max_length, padding="max_length", 

    inputs = Dataset.from_dict({"text": text_list}).map(tokenize, batched=True)
    
    return inputs


def load_roberta_path(model_name, inputs, **kwargs):

    model_path = f"models/{kwargs["segment_duration"]}s/{model_name}/{model_name}_{kwargs["segment_duration"]:g}s_epochs{kwargs["n_epochs"]}_lr{kwargs["lr"]}_drop-hidden{kwargs["hidden_dropout"]:g}_drop-attn{kwargs["attn_dropout"]:g}_drop-clf{kwargs["clf_dropout"]:g}_tune{kwargs["tune_layer"]}"
    
    return model_path

    
def load_roberta_predictor(model_name, tokenizer, inputs, **kwargs):

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
                                      logging_strategy="epoch",
                                      save_strategy="no",
                                      load_best_model_at_end=False,
                                      num_train_epochs=kwargs["n_epochs"],
                                      warmup_steps=kwargs["warmup_steps"],
                                      save_total_limit=1)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(kwargs["device"])

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "roberta" in name and f".{str(kwargs["tune_layer"])}." in name: 
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True

    trainer = CustomTrainer(model=model,
                            args=training_args,
                            train_dataset=inputs,
                            processing_class=tokenizer,
                            compute_metrics=utils.compute_metrics)

    return trainer


def load_roberta_features(model_name, model_path, speech_line_df, valence_file, max_dur=30):
    text_tokenizer = RobertaTokenizer.from_pretrained(model_name)
    text_inputs = load_text_inputs(speech_line_df["text"], text_tokenizer)

    text_model_path = utils.list_files(f"{model_path}/{max_dur}s/{model_name}")
    text_n_epochs = int(text_model_path.split("epochs")[1].split("_")[0])
    text_lr = float(text_model_path.split("lr")[1].split("_")[0])
    text_drop_hidden = float(text_model_path.split("hidden")[1].split("_")[0])
    text_drop_attn = float(text_model_path.split("attn")[1].split("_")[0])
    text_drop_clf = float(text_model_path.split("clf")[1].split("_")[0])
    text_tune_layer = int(text_model_path.split("tune")[1])
    
    text_trainer = load_roberta_predictor(model_name, text_tokenizer, text_inputs, 
                                          segment_duration=max_dur, n_epochs=text_n_epochs, lr=text_lr, 
                                          hidden_dropout=text_drop_hidden, attn_dropout=text_drop_attn, clf_dropout=text_drop_clf,
                                          tune_layer=text_tune_layer, batch_size=1, warmup_steps=500, device="mps:0")
    
    text_predictions = text_trainer.predict(text_inputs).predictions[:,0]
    
    speech_line_df["text_predicted_emotion"] = text_predictions
    speech_line_df.to_csv(valence_file)

    return speech_line_df


################### Audio-Derived Affective Valence ####################

def load_audio_inputs(audio_list, feature_extractor):

    audio_dict = {"audio": []}
    for path in audio_list:
        audio, orig_sr = librosa.load(path, sr=feature_extractor.sampling_rate)
        audio_dict["audio"] += [audio]
    
    def process(examples):
        return feature_extractor(examples["audio"], sampling_rate=feature_extractor.sampling_rate, 
                                 padding="longest", return_tensors="pt")

    inputs = Dataset.from_dict(audio_dict).map(process, batched=True)
    
    return inputs


def load_whisper_path(model_name, inputs, **kwargs):

    model_path = f"models/{kwargs["segment_duration"]}s/{model_name}/{model_name}_{kwargs["segment_duration"]}s_epochs{kwargs["n_epochs"]}_lr{kwargs["lr"]}_drop-conn{kwargs["conn_dropout"]:g}_drop-actv{kwargs["actv_dropout"]:g}_drop-attn{kwargs["attn_dropout"]:g}_tune{kwargs["tune_layer"]}"
    
    return model_path


def load_whisper_predictor(model_name, feature_extractor, inputs, **kwargs):

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
                                      num_train_epochs=kwargs["n_epochs"],
                                      warmup_steps=kwargs["warmup_steps"],
                                      save_total_limit=1)
    
    model = WhisperForAudioClassification.from_pretrained(model_path).to(kwargs["device"])

    for name, param in model.named_parameters():
        param.requires_grad = False
        if f".{kwargs["tune_layer"]}." in name:
            param.requires_grad = True
        if ".layer_norm." in name:
            param.requires_grad = True
        if "projector." in name:
            param.requires_grad = True
        if "classifier." in name:
            param.requires_grad = True
    
    trainer = CustomTrainer(model=model,
                args=training_args,
                train_dataset=inputs,
                processing_class=feature_extractor,
                compute_metrics=compute_metrics)

    return trainer


def load_whisper_features(model_name, model_path, speech_line_df, valence_file, max_dur=30):
    audio_featurizer = WhisperFeatureExtractor.from_pretrained(f"openai/{model_name}")
    audio_inputs = load_audio_inputs(speech_line_df["audio_path"], audio_featurizer)

    audio_model_path = utils.list_files(f"{model_path}/{max_dur}s/{model_name}")
    audio_n_epochs = int(audio_model_path.split("epochs")[1].split("_")[0])
    audio_lr = float(audio_model_path.split("lr")[1].split("_")[0])
    audio_drop_conn = float(audio_model_path.split("conn")[1].split("_")[0])
    audio_drop_actv = float(audio_model_path.split("actv")[1].split("_")[0])
    audio_drop_attn = float(audio_model_path.split("attn")[1].split("_")[0])
    audio_tune_layer = int(audio_model_path.split("tune")[1])
    
    audio_trainer = load_whisper_predictor(model_name, audio_featurizer, audio_inputs, 
                                           segment_duration=max_dur, n_epochs=audio_n_epochs, lr=audio_lr, 
                                           conn_dropout=audio_drop_conn, actv_dropout=audio_drop_actv, attn_dropout=audio_drop_attn,
                                           tune_layer=audio_tune_layer, batch_size=1, warmup_steps=500, device="mps:0")
    
    audio_predictions = audio_trainer.predict(audio_inputs).predictions[:,0]
    
    speech_df["audio_predicted_emotion"] = audio_predictions
    speech_df.to_csv(valence_file)
    
    return speech_line_df


####################### Grammatical NLP Features #######################

nlp = spacy.load("en_core_web_sm")

def compute_n_words(text):
    word_count = len(text.split())
    return word_count

def compute_n_person(text, person):
    person2pronouns = {
        "first_person_singular": ["i", "me", "my", "mine"],
        "first_person_plural": ["we", "us", "our", "ours"],
        "second_person_singular": ["you", "your", "yours", "yourself"],
        "second_person_plural": ["y'all", "yourselves"],
        "third_person_singular": ["he", "him", "his", "himself", "she", "her", "hers", "herself"],
        "third_person_plural": ["they", "them", "their", "theirs", "themselves"]
    }
    person_pronouns = person2pronouns[person]
    tokens = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(tokens)
    pronoun_count = len([word for word, tag in tagged_words if tag in ["PRP", "PRP$"] and word.lower() in person_pronouns])
    return pronoun_count

def compute_n_tense(text, tense):
    doc = nlp(text)
    tense_count = 0
    for sent in list(doc.sents):
        root_tag = sent.root.tag_
        if tense == "past":
            if root_tag == "VBD" or any(w.dep_ == "aux" and w.tag_ == "VBD" for w in sent.root.children):
                tense_count += 1
        if tense == "present":
            if root_tag in ["VBP", "VBZ", "VBG"] or any(w.dep_ == "aux" for w in sent.root.children):
                tense_count += 1
        if tense == "future":
            if any(w.lower_ in ["will", "shall"] for w in sent.root.children if w.dep_ == "aux"):
                tense_count += 1
    return tense_count

def compute_hedges_ratio(text):
    hedge_words = ["think", "might", "believe", "seems", "likely", "suggests"]
    hedge_pos_tags = ["MD"] # Modal verbs
    tokens = nltk.tokenize.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    detected_hedges = []
    for word, tag in tagged_tokens:
        if word.lower() in hedge_words:
            detected_hedges.append((word, "lexicon"))
        if tag in hedge_pos_tags:
            detected_hedges.append((word, "pos_tag"))
    hedges_ratio = len(detected_hedges)/ len(tagged_tokens)
    return hedges_ratio


######################## Acoustic NLP Features #########################

def compute_words_per_sec(text, duration):
    words_per_sec = len(text.split())/duration
    return words_per_sec

def compute_pause_duration_per_sec(audio_file, duration,
                              min_silence_len=200, silence_thresh=-50):
    audio = AudioSegment.from_wav(audio_file)
    silences = detect_silence(audio, 
                              min_silence_len=min_silence_len, 
                              silence_thresh=silence_thresh)
    pauses = []
    for start_ms, end_ms in silences:
        pauses.append({
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "duration_ms": end_ms - start_ms
        })
    if len(pauses) > 0:
        pause_duration = sum([pause["duration_ms"] for pause in pauses])/1000
    else:
        pause_duration = 0
    pause_duration_per_sec = pause_duration/duration
    return pause_duration_per_sec
    
def compute_jitter(audio_file, f0min=75, f0max=300):
    sound = parselmouth.Sound(audio_file)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    jitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    return jitter

def compute_shimmer(audio_file, f0min=75, f0max=300):
    sound = parselmouth.Sound(audio_file)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    shimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return shimmer
    
def compute_pitch_std(audio_file):
    sound = parselmouth.Sound(audio_file)
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array["frequency"]
    pitch_values[pitch_values==0] = np.nan
    pitch_std = np.nanstd(pitch_values)
    return pitch_std

    
######################### Combined NLP Features ########################

def generate_nlp_features(speech_line_df):

    ### Text-Derived NLP Features ###
    
    # Number of words
    speech_line_df["n_words"] = speech_line_df["text"].apply(compute_n_words)

    # Number of grammatical person words
    text_features = {}
    for feature in ["first_person_singular", 
                    "first_person_plural", 
                    "second_person_singular",
                    "third_person_singular", 
                    "third_person_plural"]:
        text_features[feature] = lambda t, c=feature: compute_n_person(t, c)
    
    # Number of tense words
    for feature in ["past", "present", "future"]:
        text_features[feature] = lambda t, c=feature: compute_n_tense(t, c)
    
    # Ratios of grammatical variants to number of words
    for name, func in text_features.items():
        speech_line_df[f"n_{name}_to_n_words"] = speech_line_df["text"].apply(func)/speech_line_df["n_words"]

    # Number of hedge words
    speech_line_df["n_hedges_to_n_words"] = speech_line_df["text"].apply(compute_hedges_ratio)

    ### Audio-Derived NLP Features ###
    
    # Words per second
    speech_line_df["words_per_sec"] = speech_line_df.apply(
        lambda row: compute_words_per_sec(row["text"], row["duration"]), axis=1)

    # Pause duration per second
    speech_line_df["pause_duration_per_sec"] = speech_line_df.apply(lambda row: compute_pause_duration_per_sec(row["audio_path"], row["duration"]), axis=1)

    # Jitter, shimmer, and standard deviation of the pitch
    audio_features = {"jitter": compute_jitter, 
                      "shimmer": compute_shimmer, 
                      "pitch_std": compute_pitch_std}
    for column, func in audio_features.items():
        speech_line_df[column] = speech_line_df["audio_path"].apply(func)
    
    return speech_line_df


########################################################################
######################## CANDIDATE GPT FEATURES ########################
########################################################################

def derive_gpt_features(speech_line_df, prompt, api_url, api_key, gpt5_headers, feature_path,
                        min_n_vars=1, max_n_vars=10, max_n_iter=3, 
                        split="train", biotypes=["low", "high"], conditions=["placebo", "mdma_high"]):

    for n_iter in range(max_n_iter):
        for condition in conditions:
            training_texts = speech_line_df[(speech_line_df["split"] == split) & (speech_line_df["condition"] == condition)][["biotype", "text"]]
            training_texts = training_texts.sample(frac=1) # Shuffle the training texts
            training_texts = "\n\n--------------------\n\n".join([f"{biotype.upper()} GROUP INPUT: {text}" for biotype, text in zip(training_texts["biotype"], training_texts["text"])])
            for biotype in biotypes:
                not_biotype = [b for b in biotypes if b != biotype][0]
                prompt_for_bioptype = prompt.replace("[BIOTYPE]", biotype).replace("[NOT_BIOTYPE]", not_biotype)
                
                for n_vars in range(min_n_vars, max_n_vars+1):
                    file_path = f"{feature_path}/gpt5/iteration_{n_iter:02d}/{condition}/{biotype}"
                    if not os.path.exists(file_path):
                        os.makedirs(f"{feature_path}/gpt5/iteration_{n_iter:02d}/{condition}/{biotype}")
                    
                    file_name = f"{file_path}/gpt5_{condition}_{biotype}_{n_vars:02d}.json"
                    if not os.path.exists(file_name):

                        prompt_for_n_vars = prompt_for_bioptype.replace("[N_VARS]", f"{n_vars:02d}") + training_texts
                    
                        payload = {"model": "gpt-5",
                                   "stream": False,
                                   "messages": [{"role": "user", "content": prompt_for_n_vars}]}
                        response = requests.post(api_url, headers=gpt5_headers, json=payload)
                        with open(file_name, "w") as file:
                            json.dump(response.json(), file, indent=4)
                        

def quantify_gpt_features(speech_line_df, prompt, api_url, api_key, gpt5_headers, feature_path,
                          min_n_vars=1, max_n_vars=10, max_n_iter=3, 
                          biotypes=["low", "high"], conditions=["placebo", "mdma_high"],
                          file_name=False):

    sessions = sorted(list(set(speech_line_df["session"])))
    for n_iter in range(max_n_iter):
        for condition in conditions:
            for biotype in biotypes:
                for n_vars in range(min_n_vars, max_n_vars+1):
                    for session in sessions:
                        df = speech_line_df[speech_line_df["session"] == session]
                        with open(f"{feature_path}/gpt5/iteration_{n_iter:02d}/{condition}/{biotype}/gpt5_{condition}_{biotype}_{n_vars:02d}.json", "r") as file:
                            differentiators_json = json.load(file)
                        differentiators = differentiators_json["choices"][0]["message"]["content"]
                        texts = "\n\n--------------------\n\n".join([f"{id}: {text}" for id, text in zip(df.index, df["text"])])
                        
                        file_path = f"{feature_path}/gpt5/iteration_{n_iter:02d}/{condition}/{biotype}/lines"
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)

                        if not file_name:
                            file_name = f"{file_path}/{condition}_{biotype}_{n_vars:02d}_lines_{session}.json"
                        if not os.path.exists(file_name):
                            
                            prompt_for_n_vars = prompt.replace("[DIFFERENTIATORS]", differentiators)
                            prompt_for_n_vars = prompt_for_n_vars.replace("[N_VARS]", f"{n_vars:02d}") + texts
                            
                            payload = {"model": "gpt-5",
                                       "stream": False,
                                       "messages": [{"role": "user", "content": prompt_for_n_vars}]}
                            response = requests.post(api_url, headers=gpt5_headers, json=payload)
                            with open(file_name, "w") as file:
                                json.dump(response.json(), file, indent=4)


def load_gpt_headers(feature_info_file):
    
    with open(feature_info_file, "r") as file:
        data_json = json.load(file)
        text = data_json["choices"][0]["message"]["content"].split("Differentiator variable names:")
        headers = text[1].split(",")
        
        if len(headers) == 1:
            headers = text[1].split("\n")
        headers = [header.strip().lower() for header in headers]
        processed_headers = []
        
        for header in headers:
            if "=" in header:
                processed_headers += [header.split("=")[1].strip()]
            elif ":" in header:
                processed_headers += [header.split(":")[1].strip()]
            else:
                processed_headers += [header]
                
        headers = [f"gpt5_{header}" for header in processed_headers if len(header) > 0]
        headers = ["gpt5_ID"] + headers
    
    return headers
    
        
def load_gpt_feature_files(feature_quant_files, speech_line_df,
                           headers=False, feature_info_file=""):

    if not headers:
        headers = load_gpt_headers(feature_info_file)
    
    gpt_df = pd.DataFrame()
    for feature_quant_file in feature_quant_files:
        if os.path.exists(feature_quant_file):
            with open(feature_quant_file, "r") as file:
                data_json = json.load(file)
                lines = data_json["choices"][0]["message"]["content"].split("\n")
                lists = [line.split(",") for line in lines]
                df = pd.DataFrame(lists[1:], columns=headers)
                df.index = df["gpt5_ID"]
                df = df[df.columns[1:]]
                df.to_csv(feature_quant_file.replace(".json", ".csv"))
                df[headers[1:]] = df[headers[1:]].astype(float)
                gpt_df = pd.concat([gpt_df, df])

    gpt_df = pd.merge(speech_line_df, gpt_df, left_index=True, right_index=True, how="inner")
    
    return gpt_df

    
def load_gpt_features(speech_line_df, feature_path, min_n_vars=1, max_n_vars=10, max_n_iter=3, 
                      biotypes=["low", "high"], conditions=["placebo", "mdma_high"]):
    gpt_dfs = {}
    for n_vars in range(min_n_vars, max_n_vars+1):
        gpt_dfs[n_vars] = {}
        for n_iter in range(max_n_iter):
            gpt_dfs[n_vars][n_iter] = {}
            for condition in conditions:
                gpt_dfs[n_vars][n_iter][condition] = {}
                for biotype in biotypes:
                    feature_info_file = f"{feature_path}/gpt5/iteration_{n_iter:02d}/{condition}/{biotype}/gpt5_{condition}_{biotype}_{n_vars:02d}.json"
                    sessions = sorted(list(set(speech_line_df["session"])))
                    feature_quant_files = [f"{feature_path}/gpt5/iteration_{n_iter:02d}/{condition}/{biotype}/lines/{condition}_{biotype}_{n_vars:02d}_lines_{session}.json" for session in sessions]
                    dif_df = load_gpt_feature_files(feature_quant_files, speech_line_df, 
                                                    feature_info_file=feature_info_file)
                    gpt_dfs[n_vars][n_iter][condition][biotype] = dif_df
                    
    return gpt_dfs


def format_gpt_title(title, width=35):
    
    title = title.replace(" and ", " & ")
    title = "\n".join(textwrap.wrap(title, width=width))

    return title


def format_gpt_description(text, biotype, width=45):
    
    text = text.replace('”', '"').replace('“', '"').replace('’', "'").replace(";", ".").replace("–", "-").replace("—", " – ")
    text = text.replace("LOW ", r"$\mathrm{NTN_{A-}}$ ").replace("HIGH ", r"$\mathrm{NTN_{A+}}$ ").replace("They", biotype.title())
    text = text.replace("Low ", r"$\mathrm{NTN_{A-}}$ ").replace("High ", r"$\mathrm{NTN_{A+}}$ ")
    sentences = re.findall(r'.+?(?:[.!?]["\']?)(?=\s|$)', text, flags=re.DOTALL)
    text = "".join(sentences[:1])
    text = "\n".join(textwrap.wrap(text, width=width))
    
    return text


def load_gpt_json(gpt_json_file):

    with open(gpt_json_file, "r") as file:
        data_json = json.load(file)
        
        text = data_json["choices"][0]["message"]["content"].split("Differentiator variable names:")
        headers = text[1].split(",")
        if len(headers) == 1:
            headers = text[1].split("\n")
        headers = [header.strip().lower() for header in headers]
        processed_headers = []
        for header in headers:
            if "=" in header:
                processed_headers += [header.split("=")[1].strip()]
            elif ":" in header:
                processed_headers += [header.split(":")[1].strip()]
            else:
                processed_headers += [header]
        var_names = [f"gpt5_{header}" for header in processed_headers if len(header) > 0]
        
        var_titles, var_descriptions = {}, {}

        lines = data_json["choices"][0]["message"]["content"].split("\n")
        lines = [line for line in lines[1:-2] if len(line.strip()) > 0]
        
        i = 0
        for line in lines:
            if i < len(var_names):
                var_name = var_names[i]
                if line[0] in [str(d) for d in range(0, 10)]:
                    var_title = line.split(":")[0][3:].strip().capitalize()
                    var_titles[var_name] = var_title
                    i += 1
        
        i = 0
        for line in lines:
            if i < len(var_names):
                if ":" in line:
                    var_name = var_names[i]
                    var_descriptions[var_name] = line.split(":")[1].strip()
                    i += 1

        gpt_json_dict = {"titles": var_titles, "descriptions": var_descriptions}
        
    return gpt_json_dict


def load_gpt_feature_info(gpt_top_params, feature_path, 
                          conditions=["placebo", "mdma_high"], biotypes=["low", "high"]):
    
    var_info = {}
    for dif_condition in conditions:
        var_info[dif_condition] = {}
        for dif_biotype in biotypes:
            n_vars = gpt_top_params[dif_condition][f"{dif_biotype}_n_vars"]
            n_iter = gpt_top_params[dif_condition][f"{dif_biotype}_n_iter"]

            gpt_json_file = f"{feature_path}/gpt5/iteration_{n_iter:02d}/{dif_condition}/{dif_biotype}/gpt5_{dif_condition}_{dif_biotype}_{n_vars:02d}.json"
            gpt_json_dict = load_gpt_json(gpt_json_file)
            var_info[dif_condition][dif_biotype] = gpt_json_dict

    gpt2title, gpt2label, gpt2description, gpt2yticks = {}, {}, {}, {}
    for condition in conditions:
        for biotype in biotypes:
            info_dict = var_info[condition][biotype]
            vars = info_dict["titles"].keys()
            gpt2title |= {var: format_gpt_title(title) for var, title in info_dict["titles"].items()}
            gpt2label |= {var: "" for var in vars}
            gpt2description |= {var: format_gpt_description(text, biotype) for var, text in info_dict["descriptions"].items()}
            gpt2yticks |= {var: [0, 0.5, 1] for var in vars}
    
    return gpt2title, gpt2label, gpt2description, gpt2yticks
    

########################################################################
######################### STATISTICAL FILTERING ########################
########################################################################

def filter_by_biotype_differences(df, stat_dfs, vars, sig_threshold=0.05, verbose=True,
                                conditions=["placebo", "mdma_high"], biotypes=["low", "high"]):

    biotype2complement = {"low": "high", "high": "low"}
    filtered_dfs = {}
    for condition in conditions:
        filtered_dfs[condition] = {}
        for biotype in biotypes:
            not_biotype = {}
            stat_df = stat_dfs[condition]
            headers = stat_df.index
            filtered_headers = []
            for header in headers[1:]:
                means = {b: float(stat_df[f"μ_{b}"].loc[header]) for b in biotypes}
                fdr = float(stat_df["FDR"].loc[header])
                if (means[biotype] > means[biotype2complement[biotype]]) and (fdr < sig_threshold):
                    filtered_headers += [header]
            filtered_dfs[condition][biotype] = df[filtered_headers]
            if verbose:
                print(f"{condition.upper():9s}  {biotype.upper():5s} BIOTYPE   SELECTED VARS = {len(filtered_headers):2d} / {len(headers):2d}")

    return filtered_dfs



########################################################################
####################### SELECTION BY CLASSIFIERS #######################
########################################################################

def train_logistic_regression(X, y, param_grid, seed=9,
                              compute_CI=False, CI_width=0.95, bootstrap_n_iter=10000, 
                              compute_p=False, permutation_n_iter=10000):

    np.random.seed(seed)
    
    clf = LogisticRegression()
    
    train_valid_index = [-1]*len(y["train"]) + [0]*len(y["valid"])
    train_valid_ps = PredefinedSplit(test_fold=train_valid_index)

    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=train_valid_ps, scoring="accuracy")
    grid_search.fit(pd.DataFrame(np.concatenate((X["train"], X["valid"])), columns=X["train"].columns), 
                    np.concatenate((y["train"], y["valid"])))

    top_clf = grid_search.best_estimator_
    
    metrics = {"p": {}, "CI_lower": {}, "CI_upper": {}}
    for split in X.keys():
        y_pred = top_clf.predict(X[split])
        metrics[split] = accuracy_score(y[split], y_pred)

        if compute_CI:
            index = range(len(y_pred))
            distribution = []
            for i in range(bootstrap_n_iter):
                index_resampled = np.random.choice(index, size=len(index), replace=True)
                distribution += [accuracy_score(y[split][index_resampled], y_pred[index_resampled])]
            CI_idx = int((1-CI_width)*bootstrap_n_iter) 
            distribution = sorted(distribution)
            metrics["CI_lower"][split] = distribution[CI_idx]
            metrics["CI_upper"][split] = distribution[-1*CI_idx]
                
        if compute_p:
            y_pred_permuted = y_pred
            distribution = []
            for i in range(permutation_n_iter):
                np.random.shuffle(y_pred_permuted)
                distribution += [accuracy_score(y[split], y_pred_permuted)]
            metrics["p"][split] = np.sum(np.array(distribution) > metrics[split]) / len(distribution)
    
    return top_clf, grid_search, metrics


def run_biotype_prediction(filt_dfs, speech_line_df, param_grid, splits=["train", "valid", "test"], 
                           biotypes=["low", "high"], conditions=["placebo", "mdma_high"], seed=9,
                           CI_width=0.95, bootstrap_n_iter=10000, permutation_n_iter=10000, verbose=True):

    condition2ids = {condition: speech_line_df[speech_line_df["condition"] == condition].index for condition in conditions}
    split2ids = {split: speech_line_df[speech_line_df["split"] == split].index for split in splits}
    biotype2ids = {biotype: speech_line_df[speech_line_df["biotype"] == biotype].index for biotype in biotypes}
    id2biotype = {id: biotype for id, biotype in zip(speech_line_df.index, speech_line_df["biotype"])}
    
    clf_dicts, clfs = {}, {}
    for condition in conditions:
        if verbose:
            print("-"*75+f"\nMODELS TRAINED ON {condition.upper()} CONDITION\n" + "-"*75)

        df = pd.merge(filt_dfs[condition]["low"], filt_dfs[condition]["high"], left_index=True, right_index=True)
        df = df[df.index.isin(condition2ids[condition])]
        vars = df.columns
        
        if len(vars) > 0:
    
            X, y = {}, {}
            for split in splits:
                df_split = df[df.index.isin(split2ids[split])].dropna()
                df_split = df_split.sample(frac=1, random_state=seed)
                X[split] = df_split[vars]
                y[split] = np.array([0 if id2biotype[id] == "low" else 1 for id in df_split.index])
            
            clf, grid_search, accuracy = train_logistic_regression(X, y, param_grid, seed=seed,
                                                                   compute_CI=True, CI_width=CI_width, 
                                                                   bootstrap_n_iter=bootstrap_n_iter, 
                                                                   compute_p=True, 
                                                                   permutation_n_iter=permutation_n_iter)
            
            clf_dict = {"condition": condition, 
                        "low_n_vars_filtered": len(filt_dfs[condition]["low"].columns),
                        "high_n_vars_filtered": len(filt_dfs[condition]["high"].columns),
                        "sum_n_vars": len(filt_dfs[condition]["low"].columns)+len(filt_dfs[condition]["high"].columns),
                        "penalty": clf.get_params()["penalty"], 
                        "C": clf.get_params()["C"], 
                        "max_iter": clf.get_params()["max_iter"]}
            
            for split in splits:
                clf_dict[f"accuracy_{split}"] = accuracy[split]
                clf_dict[f"CI_lower_{split}"] = accuracy["CI_lower"][split]
                clf_dict[f"CI_upper_{split}"] = accuracy["CI_upper"][split]
                clf_dict[f"p_{split}"] = accuracy["p"][split]
            
            if verbose:
                print(f"VARIABLES: {", ".join(vars)}\n")
                print(f"Penalty: {clf.get_params()["penalty"]}  C: {clf.get_params()["C"]}  Max Iter: {clf.get_params()["max_iter"]}  Intercept: {clf.get_params()["fit_intercept"]}\n")
                for split in splits:
                    split_df = df[df.index.isin(split2ids[split])].dropna()
                    print(f"{split.upper():6s} N low = {len(split_df.index.isin(biotype2ids["low"])):03d}  N high = {len(split_df.index.isin(biotype2ids["high"])):03d}  Accuracy = {accuracy[split]*100:6.2f}%  CI = {accuracy["CI_lower"][split]*100:4.2f}-{accuracy["CI_upper"][split]*100:4.2f}%  (p = {accuracy["p"][split]:5.4f})")
                print()
            
            clf_dicts[condition] = clf_dict
            clfs[condition] = clf
        
        else:
            clf_dicts[condition] = {}
            if verbose:
                print("No variables for training\n")

    return clf_dicts, clfs