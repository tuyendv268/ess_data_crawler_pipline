import os
import librosa
import wave
import webrtcvad
from tqdm import tqdm
import soundfile as sf
import json

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
    vad.set_mode(2)
    
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

def get_vad_results(path):
    frame_duration = 10
    sample_rate = 16000
    num_channels = 1
    sample_width = 2
    
    process_vads_result = []
    start, end = 0, 0
    offset = 0

    
    frames = load_frame(path, num_channels, sample_width, sample_rate)
    frames_vaded = get_vad_label(frames, sample_width, frame_duration, sample_rate)
    
    if frames_vaded[0].is_speech:
        start = offset
    offset = 0.1
    prev = frames_vaded[0]
    for frame in frames_vaded:
        if frame.is_speech:
            if not prev.is_speech:
                start = round(offset,3)
        else:
            if prev.is_speech:
                end = offset+frame_duration/1000
                duration = round(end-start, 3)
                process_vads_result.append(
                    {
                        "audio_filepath":path,
                        "offset":start, 
                        "duration":duration, 
                        "label":"UNK",
                        "uniq_id":path.split("/")[-1].replace(".wav",""),
                    }
                )
        
        offset += frame_duration/1000
        prev = frame
    
    return process_vads_result