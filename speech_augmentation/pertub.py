import random
import segment
import numpy as np
import soundfile as sf

class NoisePerturbation():
    def __init__(
        self,
        path,
        min_snr_db,
        max_snr_db,
        max_gain_db,
        rng=None
    ):
        self._path=path
        self._min_snr_db=min_snr_db
        self._max_snr_db=max_snr_db
        self._max_gain_db=max_gain_db  
        self._rng = random.Random() if rng == None else rng
        
    def set_path(self, path):
        self._path = path
    
    def perturb(self, data):
        print(self._path)
        noise = self.load_noise(
            path=self._path,
            sample_rate=data._sample_rate
        )
        self.perturb_with_noise(data, noise)
        
    def perturb_with_noise(self, data, noise, data_rms=None):
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db)
        if data_rms is None:
            data_rms = data.rms_db
        noise_gain_db = min(data_rms-snr_db-noise.rms_db, self._max_gain_db)
        noise.gain_db(noise_gain_db)
        
        if noise.duration > data.duration:
            start_time = self._rng.uniform(0, noise.duration - data.duration)
            if noise.duration > (start_time  + data.duration):
                noise.subsegment(start_time, start_time+data.duration)
            data._wavs += noise._wavs
        else:
            start_index = self._rng.randint(0, data._wavs.shape[0] - noise._wavs.shape[0])
            data._wavs[start_index: start_index + noise._wavs.shape[0]] += noise._wavs
            
    def load_noise(self, path, sample_rate):
        noise = segment.WavSegment.from_file(path, sample_rate)
        return noise
    
class WhiteNoisePerturbation():
    def __init__(
        self,
        min_snr_db,
        max_snr_db,
        rng=None
    ):
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._rng = np.random.RandomState() if rng is None else rng

    def perturb(self, data):
        data_rms = data.rms_db
        snr_db = self._rng.randint(self._min_snr_db, self._max_snr_db)
        noise_gain_db = data_rms-snr_db
        
        noise_signal = self._rng.randn(data._wavs.shape[0]) * (10.0 ** (noise_gain_db / 20.0))
        data._wavs += noise_signal