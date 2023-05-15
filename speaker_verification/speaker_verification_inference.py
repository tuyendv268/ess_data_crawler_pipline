import os
import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

MODEL_CONFIG = 'config/titanet-large.yaml'
config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))

verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from("titanet-large-finetune.nemo")

path2audio_file1 = 'outputs/segments/speaker_0/sub_segment_0.rttm_speaker_0.wav'
path2audio_file2 = 'outputs/segments/speaker_4/sub_segment_0.rttm_speaker_4.wav'
verification_model.verify_speakers(path2audio_file1, path2audio_file2, threshold=0.7)