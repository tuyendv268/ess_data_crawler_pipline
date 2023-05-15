from perturb import NoisePerturbation, WhiteNoisePerturbation
import os
import numpy as np
from segment import WavSegment
import random
import soundfile as sf
from tqdm import tqdm

class Augment_Pipline():
    def __init__(self, config):
        self._noise_path = config["general"]["noise_path"]
        self._data_path = config["general"]["data_path"]
        self._output_path = config["general"]["output_path"]
        self._min_snr_db = int(config["general"]["min_snr_db"])
        self._max_snr_db = int(config["general"]["max_snr_db"])
        self._max_gain_db = int(config["general"]["max_gain_db"])
        self._sample_rate = int(config["general"]["sample_rate"])
        self._rng = np.random.RandomState()
        
        self.piplines = [
            NoisePerturbation(
                path=None, 
                min_snr_db=self._min_snr_db, 
                max_snr_db=self._max_snr_db, 
                max_gain_db=self._max_gain_db,
                rng=self._rng
            ),
            WhiteNoisePerturbation(
                min_snr_db=self._min_snr_db,
                max_snr_db=self._max_snr_db,
                rng=self._rng
            )
        ]
    
    def augment(self):
        data_path = os.listdir(self._data_path)
        data_paths = [os.path.join(self._data_path, ele) for ele in data_path]
        
        for path in tqdm(data_paths):
            if ".rttm" in path:
                continue
            choice = random.randint(0, 1)
            augment = self.piplines[choice]
            if type(augment) is NoisePerturbation:
                noise_path = self.sample_noise()
                augment.set_path(noise_path)
            
            data = WavSegment.from_file(path=path, sample_rate=self._sample_rate)
            augment.perturb(data)
            
            self.save(data=data)
    
    def save(self, data):
        path = os.path.join(self._output_path, data._path.split("/")[-1])
        sf.write(path, data._wavs, samplerate=self._sample_rate)
        print("saved: ", path)
        
    def sample_noise(self):
        noises = os.listdir(self._noise_path)
        noise_path = random.choice(noises)
        noise_path = os.path.join(self._noise_path, noise_path)
        
        return noise_path