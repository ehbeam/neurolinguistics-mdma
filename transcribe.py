import utils

import os
import re
import wave
import soundfile
import pydub
import torchaudio
import librosa
import pandas as pd
import numpy as np


########################################################################
########################### PREPROCESS AUDIO ###########################
########################################################################

def convert_to_wav(mp3_file, wav_file):
    sound = pydub.AudioSegment.from_mp3(mp3_file)
    sound.export(wav_file, format="wav")


def combine_audio_files(input_files, output_file):
    data = []
    for file in input_files:
        w = wave.open(file, "rb")
        data.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
        
    output = wave.open(output_file, "wb")
    output.setparams(data[0][0])
    for i in range(len(data)):
        output.writeframes(data[i][1])
    output.close()


def reformat_audio(input_files):
    
    # Identify and reformat session files to combine
    sessions = {}
    for file in input_files:
    
        participant = re.findall(r"raw/(.*)_S", file)[0]
        session = re.findall(r"_(.*)_", file)[0].split("_")[0]
    
        if participant not in sessions.keys():
            sessions[participant] = {}
        sessions[participant][session] = []
    
        wav_file = f"{file.split(".")[0]}.WAV"
        if file.endswith("mp3"):
            convert_to_wav(file, wav_file)
        
        reformatted_file = wav_file.replace("raw", "wav")
        if not os.path.exists(reformatted_file):
            data, sample_rate = soundfile.read(wav_file)
            soundfile.write(reformatted_file, data, sample_rate)
    
    # Specify reformatted .wav files
    reformatted_path = "data/audio/wav"
    reformatted_files = [f"{reformatted_path}/{file}" for file in utils.list_files(reformatted_path)]
    
    # Specify combined .wav files
    combined_path = "data/audio/sessions"
    
    # Combine session files
    for participant, session_set in sorted(sessions.items()):
    
        for session in session_set:
            session_files = []
            
            for file in reformatted_files:
                if participant in file and session in file:
                    session_files += [file]
            session_files = sorted(session_files)
    
            combo_file = f"{combined_path}/{participant}_{session}.wav"
            if not os.path.exists(combo_file):
                combine_audio_files(session_files, combo_file)
    return sessions


def preprocess_audio(audio, file, sampling_rate=16000):
    audio = audio.set_channels(1)
    audio.export(file, format="wav")
    audio, orig_sr = librosa.load(file, sr=None)
    audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=sampling_rate)
    sf.write(file, audio, sampling_rate)
    return audio


########################################################################
###################### DIARIZE & TRANSCRIBE AUDIO ######################
########################################################################

def transcribe_audio(sessions, diarize_model, transcribe_model, verbose=True):
    transcripts = {}
    
    for participant in sorted(sessions):
        if verbose:
            print(f"Processing participant {participant}")
    
        transcripts[participant] = {}
        
        for session in sorted(list(sessions[participant].keys())):
            if verbose:
                print(f"\tProcessing session {session}")
            
            audio_file = f"data/audio/sessions/{participant}_{session}.wav"
    
            #######################
            ### Run diarization ###
            #######################
    
            diarization_file = f"data/diarization/{participant}_{session}.csv"
            
            if not os.path.exists(diarization_file):
                if verbose:
                    print("\t\tRunning diarization")
            
                waveform, sample_rate = torchaudio.load(audio_file)
                diarization = diarize_model({"waveform": waveform, "sample_rate": sample_rate})
                
                speaker_segments = []
                for segment, track, speaker in diarization.itertracks(yield_label=True):
                    speaker_segments.append({"SPEAKER": speaker, "START": segment.start, "END": segment.end})
                
                diarization_df = pd.DataFrame(speaker_segments)
                diarization_df.to_csv(diarization_file, index=None, encoding="utf-8")
            
            else:
                if verbose:
                    print("\t\tLoading diarization")
    
                diarization_df = pd.read_csv(diarization_file)
                speaker_segments = diarization_df.to_dict("records") 
    
            #########################
            ### Run transcription ###
            #########################
            
            transcript_file = f"data/transcripts/raw/{participant}_{session}.csv"
            
            if not os.path.exists(transcript_file):
                print(f"\t\tRunning transcription")
    
                audio = pydub.AudioSegment.from_file(audio_file)
                for i, segment in diarization_df.iterrows():
            
                    start = segment["START"] * 1000 # Convert to milliseconds
                    end = segment["END"] * 1000
                    speaker = int(segment["SPEAKER"].split("_")[-1])
                    
                    segment_path = f"data/audio/utterances/{participant}_{session}"
                    if not os.path.exists(segment_path):
                        os.makedirs(segment_path)
                    output_file = f"{segment_path}/{participant}_{session}_s{i:03d}_speaker{speaker:02d}.wav"
                    
                    audio_segment = pydub.AudioSegment.from_wav(audio_file)
                    audio_segment = audio_segment[start:end]
                    audio_segment.export(output_file, format="wav")
                
                audio_files = sorted([f"{segment_path}/{file}" for file in utils.list_files(segment_path)])
                
                transcript_segments = []
                for j, audio_file in enumerate(audio_files):
        
                    if j % 50 == 0:
                        if verbose:
                            print(f"\t\t\tProcessing utterance {j:03d}")
        
                    transcription = transcribe_model.transcribe(audio_file, fp16=False, length_penalty=1.0)
                    transcript_segments.append(transcription["text"].strip())
        
                transcript_df = diarization_df.copy()
                transcript_df["TEXT"] = transcript_segments
                transcript_df.to_csv(transcript_file, index=None, encoding="utf-8")
    
                transcripts[participant][session] = transcript_df
    
            else:
                if verbose:
                    print(f"\t\tTranscription file found")
    return transcripts


########################################################################
######################### PROCESS TRANSCRIPTS ##########################
########################################################################

def combine_by_speaker(input_path, output_path):
    input_files = sorted(utils.list_files(input_path))

    output_path = f"{output_path}/speakers"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for file_name in input_files:
    
        input_file =  f"{input_path}/{file_name}"
        output_file = f"{output_path}/{file_name}"
        
        transcript_df = pd.read_csv(input_file, encoding="latin-1")
        transcript_df = transcript_df.dropna(subset=["SPEAKER"])
        transcript_df = transcript_df[["SPEAKER", "START", "END", "TEXT"]]
        transcript_df["TEXT"] = transcript_df["TEXT"].astype(str)
        
        groups = transcript_df["SPEAKER"].ne(transcript_df["SPEAKER"].shift()).cumsum()
        transcript_df = transcript_df.groupby(groups, as_index=False).agg({"SPEAKER": "first", "START": "first", "END": "last", "TEXT": " ".join})
        transcript_df.to_csv(output_file, index=None, encoding="latin-1")


def combine_by_window(input_path, main_output_path, transcribe_model,
                      sampling_rate=16000, min_dur=1, max_durs=[30,20,10], verbose=True):
    
    transcript_dfs = {}
    for max_dur in max_durs: # seconds

        if verbose:
            print(f"Processing {max_dur}-second segments")
        
        input_files = sorted(utils.list_files(input_path))

        output_path = f"{main_output_path}/{int(max_dur)}s"
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        transcript_dfs[max_dur] = {}
        for file_name in input_files:

            if verbose:
                print(f"\tProcessing {file_name}")
                
            input_file =  f"{input_path}/{file_name}"
            output_file = f"{output_path}/{file_name}"
            session = os.path.splitext(file_name)[0]
            
            if not os.path.exists(output_file):
        
                audio_path = f"data/audio/segmented/{max_dur}s/{session}"
                if not os.path.exists(audio_path):
                    os.makedirs(audio_path)
                audio_segment_input = pydub.AudioSegment.from_file(f"data/audio/sessions/{session}.wav")
                
                df = pd.read_csv(input_file, encoding="latin-1")
                df = df.dropna(subset=["SPEAKER"])
                df = df[["SPEAKER", "START", "END", "TEXT"]]
                df["START"] = df["START"].astype(float)
                df["END"] = df["END"].astype(float)
                df["TEXT"] = df["TEXT"].astype(str)
                df["DURATION"] = df["END"] - df["START"]
    
                # Filter out lines less than the minimum duration filter
                df = df[df["DURATION"] > min_dur]
                
                # Group lines by speaker within a rolling window
                transcript_duration = int(df["END"].values[-1])
                window_duration = int(max_dur)
                grouped_df = pd.DataFrame()
                for window_start in range(0, transcript_duration, window_duration):
                    window_df = df[(df["START"] > window_start) & (df["START"] < (window_start + window_duration))]
                    groups = window_df["SPEAKER"].ne(window_df["SPEAKER"].shift()).cumsum()
                    window_df = window_df.groupby(groups, as_index=False).agg({"SPEAKER": "first", "START": "first", "END": "last", "TEXT": " ".join})
                    grouped_df = pd.concat([grouped_df, window_df])
                grouped_df["DURATION"] = grouped_df["END"] - grouped_df["START"]
                grouped_df = grouped_df[["SPEAKER", "START", "END", "DURATION", "TEXT"]]
                grouped_dict = grouped_df.to_dict(orient="records")
            
                truncated_dict = []
                for i, line in enumerate(grouped_dict):
                    line["LINE"] = i
                    
                    if line["DURATION"] <= max_dur:
    
                        line["SEGMENT"] = 0
                        
                        id = f"{session}_line{i:03d}_seg{0:02d}"
                        audio_output_file = f"{audio_path}/{id}_{line["SPEAKER"].lower()}.wav"
                        line["ID"] = f"{session}_line{i:03d}_seg{0:02d}"
                        line["AUDIO_PATH"] = audio_output_file
                        
                        audio_segment = audio_segment_input[(line["START"]*1000):(line["END"]*1000)]
                        audio_segment = preprocess_audio(audio_segment, audio_output_file, sampling_rate=sampling_rate)
                        
                        truncated_dict += [line]
                    
                    else: 
                        for j in range(int(line["DURATION"]/max_dur)+1):
                            
                            start_segment = line["START"] + (max_dur*j)
                            end_segment = min(start_segment + max_dur, line["END"])
                            dur = end_segment-start_segment
                            
                            if dur > min_dur:
    
                                id = f"{session}_line{i:03d}_seg{j:02d}"
                                audio_output_file = f"{audio_path}/{id}_{line["SPEAKER"].lower()}.wav"
                                audio_segment = audio_segment_input[(start_segment*1000):(end_segment*1000)]
                                audio_segment = preprocess_audio(audio_segment, audio_output_file, sampling_rate=16000)
                
                                transcription = transcribe_model.transcribe(audio_output_file, fp16=False, length_penalty=1.0)
                                transcription = str(transcription["text"].strip())
                
                                truncated_dict += [{"SPEAKER": line["SPEAKER"], "START": start_segment, "END": end_segment, 
                                                    "DURATION": dur, "TEXT": transcription, "LINE": i, "SEGMENT": j,
                                                    "ID": id, "AUDIO_PATH": audio_output_file}]

                truncated_df = pd.DataFrame(truncated_dict)
                truncated_df = truncated_df[["ID", "LINE", "SEGMENT", "START", "END", "DURATION", "SPEAKER", "AUDIO_PATH", "TEXT"]]
                truncated_df.to_csv(output_file, index=None)

            else:
                truncated_df = pd.read_csv(output_file)
                
            transcript_dfs[max_dur][session] = truncated_df 
            
    return transcript_dfs


def combine_transcripts2df(transcript_dfs, max_dur=30):
    combined_df = pd.DataFrame()
    for session, transcript_df in transcript_dfs[max_dur].items():
        transcript_df.index = transcript_df["ID"]
        transcript_df["SESSION"] = [session]*len(transcript_df)
        transcript_df["PARTICIPANT"] = [int(session.split("_")[0].split("P")[1])]*len(transcript_df)
        transcript_df = transcript_df[["PARTICIPANT", "SESSION"] + [c for c in transcript_df.columns if c not in ["PARTICIPANT", "SESSION"]]]
        combined_df = pd.concat([combined_df, transcript_df])
    return combined_df


def combine_lines2sessions(line_df):
    session_df = line_df.groupby("session").agg(participant=("participant", "first"), 
                                                session=("session", "first"), 
                                                biotype=("biotype", "first"), 
                                                condition=("condition", "first"), 
                                                text=("text", " ".join),
                                                duration=("duration", "sum")
                                               ).copy()
    return session_df
    

def filter_line_df(line_df, speaker_filter=False, min_dur_filter=False):
    if speaker_filter:
        line_df = line_df[line_df["SPEAKER"] == speaker_filter]
    if min_dur_filter:
        line_df = line_df[line_df["DURATION"] >= min_dur_filter]
    return line_df


########################################################################
################### MERGE TRANSCRIPTS & STUDY DATA #####################
########################################################################

def load_study_data(speech_df, demo_df):
    session2participant, session2biotype, session2condition = {}, {}, {}
    for subject, biotype, placebo in zip(demo_df["Subjects"], demo_df["Group"], demo_df["Placebo"]):
        if str(placebo) != "nan":
            session = f"P{subject:03d}_{placebo}"
            session2participant[session] = session.split("_")[0]
            session2biotype[session] = biotype
            session2condition[session] = "placebo"
    for subject, biotype, mdma in zip(demo_df["Subjects"], demo_df["Group"], demo_df["MDMA_High"]):
        if str(mdma) != "nan":
            session = f"P{subject:03d}_{mdma}"
            session2participant[session] = session.split("_")[0]
            session2biotype[session] = biotype
            session2condition[session] = "mdma_high" 

    speech_df.columns = [c.lower() for c in speech_df.columns]
    speech_df = speech_df[speech_df["session"].isin(session2condition.keys())]
    speech_df["participant"] = [session2participant[session] for session in speech_df["session"]]
    speech_df["biotype"] = [session2biotype[session] for session in speech_df["session"]]
    speech_df["condition"] = [session2condition[session] for session in speech_df["session"]]

    study_vars = ["participant", "condition", "biotype"]
    speech_df = speech_df[study_vars + [c for c in speech_df.columns if c not in study_vars]]
    return speech_df


def load_blocks(speech_line_df, n_blocks=5, min_dur=10):

    sessions = sorted(list(set(speech_line_df["session"])))
    speech_block_line_df, speech_block_df = pd.DataFrame(), pd.DataFrame()
    for session in sessions:
        session_df = speech_line_df[speech_line_df["session"] == session].copy()
        dur_total = session_df["duration"].sum()
        dur_block = dur_total/n_blocks
        
        running_dur = 0
        running_durs = []
        for dur in session_df["duration"]:
            running_dur += dur
            running_durs += [running_dur]
        session_df["running_duration"] = running_durs

        for block_i in range(n_blocks):
            start_time = block_i * dur_block
            end_time = start_time + dur_block
            session_block_i_df = session_df[(session_df["running_duration"] > start_time) & (session_df["running_duration"] <= end_time)].copy()
            session_block_i_df["block"] = [block_i+1] * len(session_block_i_df)
            speech_block_line_df = pd.concat([speech_block_line_df, session_block_i_df])

            if len(session_block_i_df) > 0:
                dur_block_total = np.sum(session_block_i_df["duration"])
                block_agg_df = session_block_i_df.groupby("session").agg(session=("session", "first"),
                                                                         block=("block", "first"),
                                                                         biotype=("biotype", "first"), 
                                                                         duration=("duration", "sum"),
                                                                         text=("text", " ".join)
                                                                        )
                speech_block_df = pd.concat([speech_block_df, block_agg_df])

    return speech_block_df, speech_block_line_df


def load_index(line_df, n_blocks=5):
    index = []
    sessions = dict.fromkeys(line_df["session"])
    for session in sessions:
        df = line_df[line_df["session"] == session]
        for block in range(1, n_blocks+1):
            block_lines_df = df[df["block"] == block]
            index += [f"{session}_B{block}_L{i}" for i in range(1, len(block_lines_df)+1)]
    line_df["line_id"] = index
    line_df.index = index
    return line_df

