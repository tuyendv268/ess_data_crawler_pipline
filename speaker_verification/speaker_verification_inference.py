import os
import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

MODEL_CONFIG = 'config/titanet-large.yaml'
config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))

verification_model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from("outputs/ckpts/TitaNet-L.nemo")
# verification_model = nemo_asr.models.EncDecSpeakerLabelModel.load_from_checkpoint(
#     "outputs/ckpts/TitaNet-L--val_loss=1.0419-epoch=45.ckpt",
#     map_location="cpu",
#     hparams_file="config/hparams.yml")

path2audio_file1 = '/home/tuyendv/Desktop/codes/ess_data_crawler_pipline/outputs/speaker_diarization/speaker_0/wav_0.rttm_speaker_0_3.wav'
path2audio_file2 = '/home/tuyendv/Desktop/codes/ess_data_crawler_pipline/outputs/speaker_diarization/speaker_0/wav_0.rttm_speaker_0_25.wav'
verification_model.verify_speakers(path2audio_file1, path2audio_file2, threshold=0.7)