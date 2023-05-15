import pytorch_lightning as pl
from nemo.collections.asr.models.msdd_models import EncDecDiarLabelModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf
import os

config = OmegaConf.load('config/msdd_5scl_15_05_50Povl_256x3x32x2.yaml')

config.model.train_ds.manifest_filepath = '../outputs/diar_datas/train/msdd_data.50step.json'
config.model.validation_ds.manifest_filepath = '../outputs/diar_datas/valid/msdd_data.50step.json'
config.model.test_ds.manifest_filepath = '../outputs/diar_datas/train/test/msdd_data.50step.json'

config.batch_size=1
config.model.emb_batch_size=0
config.model.train_ds.emb_dir="../outputs/embeddings/train" 
config.model.validation_ds.emb_dir="../outputs/embeddings/valid" 
config.model.test_ds.emb_dir="../outputs/embeddings/test" 
config.model.diarizer.speaker_embeddings.model_path="titanet_large"
config.trainer.max_epochs = 20
config.trainer.strategy = None


trainer = pl.Trainer(**config.trainer)
exp_manager(trainer, config.get("exp_manager", None))
msdd_model = EncDecDiarLabelModel(cfg=config.model, trainer=trainer)

trainer.fit(msdd_model)
# msdd_model.save_to("msdd.nemo")