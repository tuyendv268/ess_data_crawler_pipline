import configparser
from pipline import Augment_Pipline

path = "agument_config.ini"
config = configparser.ConfigParser()
config.read(path)

pipline = Augment_Pipline(config)

pipline.augment()