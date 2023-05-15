import os
import time
import json
import librosa
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment
from omegaconf import OmegaConf
import speaker_diarization.reclustering_speaker_diarization_inference as reclustering_speaker_diarization_inference
import configparser
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(wav_path, format='wav')
    print(f'saved: {wav_path}')
    return wav_path

def segment_audio_by_duration(segment_dir_path, wav_path, duration):
    wav, sr = librosa.load(wav_path, sr=16000)
    # clear old wav segment
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
    infer_config.read("conf/config.cfg")
    
    # prepare data for inference
    mp3_path = "inputs/Tổng_Đài_Thông_Minh_Giúp_Tăng_Gấp_Đôi_Năng_Suất_Nuôi_Kỳ_Vọng_Đạt_100_Tỷ_Doanh_Thu_Và_Cái_Kết.mp3"
    msdd_data_path = infer_config["path"]["manifest_filepath"]
    segment_dir_path = infer_config["path"]["segment_dir_path"]
    duration=int(infer_config["general"]["duration"])
    # prepare data for inference
    
    prepare_data_for_inference(mp3_path, segment_dir_path, msdd_data_path, duration)
    # --------------------------
    
    ROOT = os.getcwd()
    conf_dir = os.path.join(ROOT,'conf')
    os.makedirs(conf_dir, exist_ok=True)
    
    pretrained_vad_path = infer_config["path"]["pretrained_vad_path"].replace("'","")
    pretrained_speaker_model = infer_config["path"]["pretrained_speaker_model"]

    MODEL_CONFIG = os.path.join(conf_dir,'diar_infer.yaml')

    config = OmegaConf.load(MODEL_CONFIG)
    config.diarizer.manifest_filepath = infer_config["path"]["manifest_filepath"]
    # config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
    # config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
    # config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
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
    # config.diarizer.vad.parameters.pad_offset = -0.05
    
    # config.diarizer.msdd_model.parameters.sigmoid_threshold = [0.7, 1.0]
    system_vad_msdd_model = NeuralDiarizer(cfg=config)

    start = time.time()
    # print("------------- diarize --------------")
    system_vad_msdd_model.diarize()
    # print("------------- global clustering --------------")

    num_sample_per_cluster = int(infer_config["general"]["num_sample_per_cluster"])
    embedding_path = infer_config["path"]["embedding_path"]
    clusters_pred_path = infer_config["path"]["clusters_pred_path"]

    input_rttm_path = infer_config["path"]["input_rttm_path"]
    output_rttm_path = infer_config["path"]["output_rttm_path"]


    embs_and_timestamps_path = "embs_and_timestamps.pkl"
    predicted_rttm_path = "outputs/pred_rttms"
    subsegment_labels_path = "outputs/speaker_outputs/subsegments_scale4_cluster.label"
    number_embedding_per_spk = 512
    duration=600.0
    
    """
    
    """
    
    reclustering_speaker_diarization_inference.global_clustering(
        embs_and_timestamps_path=embs_and_timestamps_path,
        predicted_rttm_path=input_rttm_path,
        output_rttm_dir=output_rttm_path,
        subsegment_labels_path=subsegment_labels_path,
        number_embedding_per_spk=number_embedding_per_spk,
        duration=duration
    )
    
    reclustering_speaker_diarization_inference.convert_rttms_to_segments("inputs", "outputs/segments")
    reclustering_speaker_diarization_inference.get_silence_from_rttms("inputs", "outputs/segments")

    end = time.time()
    
    print("total time: ", end-start)
