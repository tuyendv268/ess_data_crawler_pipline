from pydub import AudioSegment
import librosa
import numpy as np

class WavSegment():
    def __init__(self, wavs, path, target_sample_rate=None) -> None:
        self._path = path
        original_sample_rate = wavs.frame_rate
        wavs = self.convert_int_to_float32(wavs)
        
        if target_sample_rate is not None and target_sample_rate != original_sample_rate:
            self._sample_rate = target_sample_rate
            self._wavs = librosa.core.resample(wavs, orig_sr=original_sample_rate, target_sr=target_sample_rate)
        else:
            self._sample_rate = original_sample_rate
            self._wavs = wavs
            
    
    @classmethod
    def from_file(cls, path, sample_rate):
        wavs = AudioSegment.from_file(path)
        return cls(
            wavs,
            path,
            sample_rate
        )
    def gain_db(self, gain):
        self._wavs *= 10 ** (gain / 20.0)
    
    def subsegment(self, start_time, end_time):
        start_sample = int(round(self._sample_rate * start_time))
        end_sample = int(round(self._sample_rate * end_time))
        self._wavs = self._wavs[start_sample:end_sample]

    @staticmethod
    def convert_int_to_float32(wavs):
        wav_sample = wavs.get_array_of_samples()
        sample_width = wavs.sample_width
        wav_array = np.array(wav_sample) * 1.0/float(1 << (8*sample_width-1))
        return wav_array
    
    @property
    def rms_db(self):
        mean_sqr = np.mean(self._wavs ** 2, axis=0)
        return 10 * np.log10(mean_sqr)
    
    @property
    def _num_samples(self):
        return self._wavs.shape[0]
    
    @property
    def duration(self):
        return self._num_samples / float(self._sample_rate)