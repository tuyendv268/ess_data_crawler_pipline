import os
import glob
import subprocess
import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import torch
import pytorch_lightning as pl
from nemo.utils.exp_manager import exp_manager

MODEL_CONFIG = 'config/titanet-large.yaml'
config = OmegaConf.load(MODEL_CONFIG)

print("Trainer config - \n")
print(OmegaConf.to_yaml(config.trainer))

data_dir = "datas"
train_manifest = os.path.join(data_dir,'train.json')
validation_manifest = os.path.join(data_dir,'valid.json')
test_manifest = os.path.join(data_dir,'test.json')

config.model.train_ds.manifest_filepath = train_manifest
config.model.validation_ds.manifest_filepath = validation_manifest
config.model.decoder.num_classes = 652
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
config.trainer.devices = 1
config.trainer.accelerator = accelerator
config.trainer.max_epochs = 10
config.trainer.strategy = None
config.model.train_ds.augmentor=None

trainer = pl.Trainer(**config.trainer)
log_dir = exp_manager(trainer, config.get("exp_manager", None))
print(log_dir)
speaker_model = nemo_asr.models.EncDecSpeakerLabelModel(cfg=config.model, trainer=trainer)

trainer.fit(speaker_model)