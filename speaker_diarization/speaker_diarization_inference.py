import os
import time
import json
import librosa
from tqdm import tqdm
import pandas as pd
import soundfile as sf
from pydub import AudioSegment
from omegaconf import OmegaConf
import configparser
import sys
sys.path.append("../nemo")
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

def convert_rttm_to_wav(rttm_path, wav_path, output_path):
    os.system(f"rm -r {output_path}/*")
    for rttm_file in os.listdir(rttm_path):
        abs_rttm_path = os.path.join(rttm_path, rttm_file)
        abs_wav_path = os.path.join(wav_path, rttm_file.replace(".rttm", ".wav"))
        
        header = ["type", "wav_name", "o1", "offset", "duration", "o2", "o3", "speaker","o4", "o5"]
        rttm = pd.read_csv(abs_rttm_path, sep="\s+", names=header)
        wav, sr = librosa.load(abs_wav_path, sr = 16000)

        spk2utt = {}
        for index in rttm.index:
            if rttm["speaker"][index] not in spk2utt:
                start = int(rttm["offset"][index] * sr)
                end = int((rttm["offset"][index] + rttm["duration"][index]) * sr)
                spk2utt[rttm["speaker"][index]] = [wav[start:end]]
            else:
                start = int(rttm["offset"][index] * sr)
                end = int((rttm["offset"][index] + rttm["duration"][index]) * sr)
                spk2utt[rttm["speaker"][index]].append(wav[start:end])

        for spk, wavs in spk2utt.items():
            spk_wavs = []
            tmp_path = os.path.join(output_path, spk)
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            for i, wav in enumerate(wavs):
                abs_path = os.path.join(tmp_path, f"{rttm_file}_{spk}_{i}.wav")
                sf.write(abs_path, wav, sr)
            print("saved: ", abs_path)
                
def get_silence_from_rttm(rttm_dir, wav_dir, output_path):
    rttm_names = ["type", "wav_segment", "o1", "offset", "duration","o2", "o3",  "cluster_label", "o4", "o5"]
    sample_rate = 16000

    for file in os.listdir(rttm_dir):
        rttm_file = os.path.join(rttm_dir, file)
        wav_file = os.path.join(wav_dir, file.replace(".rttm", ".wav"))
        
        rttm_df = pd.read_csv(rttm_file, sep="\s+", names=rttm_names)
        wav, _ = librosa.load(wav_file, sr=sample_rate)
        
        start, offset = 0, 0
        wavs = []
        for index in rttm_df.index:
            offset = rttm_df["offset"][index]
            duration = rttm_df["duration"][index]
            wavs += list(wav[int(start*sample_rate):int(offset*sample_rate)])
            start=offset + duration
        sf.write(f'{output_path}/{file}_silence.wav', wavs, samplerate=sample_rate)
        print(f'saved: {output_path}/{file}_silence.wav')
    
def prepare_msdd_data_for_inference(input_path, output_path):
    with open(output_path, "w", encoding="utf-8") as tmp:
        content = ""
        for _file in os.listdir(input_path):
            abs_path = os.path.join(input_path, _file)
            input_sample = {"audio_filepath": f"{abs_path}", "offset": 0, "duration": None, "label": "infer", "text": "-", "num_speakers": None, "rttm_filepath": None, "uem_filepath": None, "ctm_filepath": None}
            json_obj = json.dumps(input_sample, ensure_ascii=False)
            content += json_obj + "\n"
        tmp.write(content)

def diarize(model_config, infer_config, input_path, output_path):
    msdd_data_path = infer_config["path"]["manifest_filepath"]
    prepare_msdd_data_for_inference(input_path, msdd_data_path)

    config.diarizer.manifest_filepath = msdd_data_path
    msdd_model = NeuralDiarizer(cfg=model_config)

    msdd_model.diarize()

if __name__ == "__main__" :
    infer_config = configparser.ConfigParser()
    infer_config.read("config/config.cfg")

    ROOT = os.getcwd()
    MODEL_CONFIG = infer_config['config']['infer_config']

    config = OmegaConf.load(MODEL_CONFIG)
    config.diarizer.manifest_filepath = infer_config["path"]["manifest_filepath"]
    output_dir = os.path.join(ROOT, 'outputs')
    config.diarizer.out_dir = output_dir
    config.diarizer.speaker_embeddings.model_path = infer_config["path"]["pretrained_speaker_model"]
    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers=False
    config.diarizer.vad.parameters.pad_offset = -0.05
    msdd_model_path = infer_config["path"]["msdd_model_path"].replace("'","")
    config.diarizer.msdd_model.model_path = msdd_model_path

    config.diarizer.vad.model_path = infer_config["path"]["pretrained_vad_path"]

    input_path = "/home/tuyendv/Desktop/codes/ess_data_crawler_pipline/datas/small"
    output_path = "/home/tuyendv/Desktop/codes/ess_data_crawler_pipline/outputs/speaker_diarization"

    diarize(
        model_config=config,
        infer_config=infer_config,
        input_path=input_path,
        output_path=output_path
    )
    
    convert_rttm_to_wav(
        rttm_path="outputs/pred_rttms",
        wav_path=input_path, 
        output_path="outputs/segments")
    
    get_silence_from_rttm(
        rttm_dir="outputs/pred_rttms", 
        wav_dir=input_path,
        output_path="outputs/segments")
