import os
import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

MODEL_CONFIG = 'config/titanet-large.yaml'
config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))

verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from("outputs/ckpts/TitaNet-L.nemo")

path2audio_file1 = '/home/tuyendv/Desktop/codes/ess_data_crawler_pipline/speaker_diarization/outputs/segments/speaker_2/sub_segment_0.rttm_speaker_2_0.wav'
path2audio_file2 = '/home/tuyendv/Desktop/codes/ess_data_crawler_pipline/speaker_diarization/outputs/segments/speaker_3/sub_segment_0.rttm_speaker_3_3.wav'
verification_model.verify_speakers(path2audio_file1, path2audio_file2, threshold=0.7)