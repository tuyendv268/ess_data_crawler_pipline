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

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(wav_path, format='wav')
    print(f'saved: {wav_path}')
    return wav_path
def convert_rttms_to_segments(input_path, output_path):
    rttm_path = f'{input_path}/rttms'
    wav_path = f'{input_path}/wavs'
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
                spk2utt[rttm["speaker"][index]] = []
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
                # spk_wavs += list(wav)
                abs_path = os.path.join(tmp_path, f"{rttm_file}_{spk}_{i}.wav")
                sf.write(abs_path, wav, sr)
            print("saved: ", abs_path)
                
def get_silence_from_rttms(input_path, output_path):
    rttm_dir = f'{input_path}/rttms'
    wav_dir = f'{input_path}/wavs'
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

def segment_audio_by_duration(segment_dir_path, wav_path, duration):
    wav, sr = librosa.load(wav_path, sr=16000)
    os.system(f'rm -r {segment_dir_path}/*')
     # minute
    segment_length = duration * 60
    step = int(segment_length * sr)
    for offset, index in enumerate(tqdm(range(0, len(wav), step))):
        tmp_path = f'{segment_dir_path}/sub_segment_{offset}.wav'
        sf.write(tmp_path, wav[index:index+step], sr)
    
def prepare_msdd_data_for_inference(segment_dir_path, output_path):
    with open(output_path, "w", encoding="utf-8") as tmp:
        content = ""
        for file in os.listdir(segment_dir_path):
            abs_path = os.path.join(segment_dir_path, file)
            input_sample = {"audio_filepath": f"{abs_path}", "offset": 0, "duration": None, "label": "infer", "text": "-", "num_speakers": None, "rttm_filepath": None, "uem_filepath": None, "ctm_filepath": None}
            json_obj = json.dumps(input_sample, ensure_ascii=False)
            content += json_obj + "\n"
        tmp.write(content)

def prepare_data_for_inference(mp3_path, segment_dir_path, msdd_data_path, duration):
        wav_path = "inputs/temp.wav"
        if mp3_path.endswith(".mp3"):
            wav_path = convert_mp3_to_wav(mp3_path, wav_path)
            print("convert mp3 to wav")
        else:
            wav_path = mp3_path
        segment_audio_by_duration(segment_dir_path, wav_path, duration)
        prepare_msdd_data_for_inference(segment_dir_path, msdd_data_path)

if __name__ == "__main__" :
    infer_config = configparser.ConfigParser()
    infer_config.read("config/config.cfg")
    
    mp3_path = "/home/tuyendv/Desktop/codes/ess_data_crawler_pipline/datas/raws/Đừng_nói_khi_yêu/Đừng nói khi yêu tập 1  Chị gái giàu có bị người tình đẹp trai và em thư ký cắm sừng thành tuần lộc.wav"
    msdd_data_path = infer_config["path"]["manifest_filepath"]
    segment_dir_path = infer_config["path"]["segment_dir_path"]
    duration=int(infer_config["general"]["duration"])
    
    prepare_data_for_inference(mp3_path, segment_dir_path, msdd_data_path, duration)
    # --------------------------
    
    ROOT = os.getcwd()
    conf_dir = os.path.join(ROOT,'config')
    os.makedirs(conf_dir, exist_ok=True)
    
    pretrained_vad_path = infer_config["path"]["pretrained_vad_path"].replace("'","")
    pretrained_speaker_model = infer_config["path"]["pretrained_speaker_model"]

    MODEL_CONFIG = infer_config['config']['infer_config']

    config = OmegaConf.load(MODEL_CONFIG)
    config.diarizer.manifest_filepath = infer_config["path"]["manifest_filepath"]
    config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
    config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
    config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    config.num_workers = 1

    output_dir = os.path.join(ROOT, 'outputs')
    os.system(f'rm -r {os.path.join(output_dir, "pred_rttms")}/*')
    config.diarizer.out_dir = output_dir

    config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    msdd_model_path = infer_config["path"]["msdd_model_path"].replace("'","")
    
    config.diarizer.msdd_model.model_path = msdd_model_path
    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers=False

    config.diarizer.vad.model_path = pretrained_vad_path
    # config.diarizer.vad.parameters.onset = 0.8
    # config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    
    # config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0]
    system_vad_msdd_model = NeuralDiarizer(cfg=config)

    start = time.time()
    system_vad_msdd_model.diarize()

    num_sample_per_cluster = int(infer_config["general"]["num_sample_per_cluster"])
    embedding_path = infer_config["path"]["embedding_path"]
    clusters_pred_path = infer_config["path"]["clusters_pred_path"]

    input_rttm_path = infer_config["path"]["input_rttm_path"]
    output_rttm_path = infer_config["path"]["output_rttm_path"]
    
    os.system(f"cp -r {input_rttm_path}/* {output_rttm_path}")
    # reclustering.global_clustering(input_rttm_path, output_rttm_path, embedding_path, clusters_pred_path, num_sample_per_cluster)
    convert_rttms_to_segments("inputs", "outputs/segments")
    get_silence_from_rttms("inputs", "outputs/segments")
    # print("------------- done ---------------")
    end = time.time()
    
    print("total time: ", end-start)
