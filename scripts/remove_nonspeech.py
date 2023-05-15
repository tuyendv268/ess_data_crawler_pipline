"""
    Remove non speech in wav file
"""

import os
import librosa
import wave
import webrtcvad
from tqdm import tqdm
import soundfile as sf


def load_frame(path, num_channels, sample_width, sample_rate):
    with wave.open(path, mode="rb") as wf:
        nchannels = wf.getnchannels()
        if nchannels != num_channels:
            print(f"num_channels must be {num_channels}")
        # print(f"nchannels: {nchannels}")
        sampwidth = wf.getsampwidth()
        if sampwidth != sample_width:
            print(f"sample width must be {sample_width}")
            # assert sampwidth == sample_width
        # print(f"sample width : {sampwidth}")
        frame_rate = wf.getframerate()
        if frame_rate != sample_rate:
            print(f"sample_rate must be {sample_rate}")
        # print(f"sample rate: {sample_rate}")
        frames = wf.readframes(wf.getnframes())
    return frames

class Frame:
    def __init__(self, start, end, is_speech, sample_rate, sample_width):
        self.start = start/(sample_rate * sample_width)
        self.end = end/(sample_rate * sample_width)
        self.is_speech = is_speech
        
    def to_string(self):
        return f"start: {self.start}\tend: {self.end}\t{self.is_speech}"


def get_vad_label(frames, sample_width, frame_duration, sample_rate):
    vad = webrtcvad.Vad()
    vad.set_mode(1)
    
    frames_vaded = []
    step = sample_width * int(frame_duration * sample_rate / 1000)
    length = len(frames)
    for index in range(0, length, step):
        frame = frames[index: index + step]
        
        if index + step >= length:
            break
        is_speech = vad.is_speech(frame, sample_rate)
            
        frames_vaded.append(
            Frame(
                start = index,
                end = index + step,
                is_speech=is_speech,
                sample_rate=sample_rate,
                sample_width=sample_width
                )
            )
    
    return frames_vaded

def remove_non_speech(path):
    frame_duration = 20
    sample_rate = 16000
    num_channels = 1
    sample_width = 2
    
    frames = load_frame(path, num_channels, sample_width, sample_rate)
    frames_vaded = get_vad_label(frames, sample_width, frame_duration, sample_rate)
    
    speechs, non_speechs = [], []

    wavs, _ = librosa.load(path, sr = sample_rate)
    if wavs.shape[0] != len(frames)/sample_width:
        return [], []

    for index, frame in enumerate(frames_vaded):
        if frame.is_speech:
            speechs += list(wavs[int(frame.start*sample_rate):int(frame.end*sample_rate)])
        else:
            non_speechs += list(wavs[int(frame.start*sample_rate):int(frame.end*sample_rate)])
    
    return speechs, non_speechs

def preprocess_wav(inp_paths, out_path):
    for folder in tqdm(inp_paths):
        wav_files = os.listdir(folder)
        
        out_folder = os.path.join(out_path, folder.split("/")[-1])
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        for path in wav_files:
            path = os.path.join(folder, path)
            speechs, non_speechs = remove_non_speech(path)
            if len(speechs) == 0:
                continue
            
            wav_name = path.split("/")[-1]
            out_wav = os.path.join(out_folder, wav_name)
            sf.write(out_wav.replace(".wav", "_speech.wav"), speechs, 16000, subtype='PCM_16')
            sf.write(out_wav.replace(".wav", "_non_speech.wav"), non_speechs, 16000, subtype='PCM_16')

if __name__ == "__main__":     
    base_path = 'youtube_datas'
    inp_paths = os.listdir(base_path)
    inp_paths = [os.path.join(base_path, path) for path in inp_paths]
    
    out_path = "removed_nonspeech_datas"
    preprocess_wav(inp_paths, out_path)