import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.io import wavfile


def softmax(x, T):
    
    x = x/T
    orig_shape = x.shape
    if len(x.shape) > 1:
        x = np.exp(x-np.max(x,axis=-1).reshape(-1,1))
        x = x / np.sum(x, axis= -1,keepdims=True)
    else:
        x = np.exp(x-np.max(x,axis=-1))
        x = x / np.sum(x,keepdims=True)

    assert x.shape == orig_shape
    return x

def wavefile_to_waveform(wav_file, features_type):
    data, sr = sf.read(wav_file)
    if features_type == 'vggish':
        tmp_name = str(int(np.random.rand(1)*1000000)) + '.wav'
        sf.write(tmp_name, data, sr, subtype='PCM_16')
        sr, wav_data = wavfile.read(tmp_name)
        os.remove(tmp_name)
        # sr, wav_data = wavfile.read(wav_file) # as done in VGGish Audioset
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  
    # at least one second of samples, if not repead-pad
    src_repeat = data
    while (src_repeat.shape[0] < sr): 
        src_repeat = np.concatenate((src_repeat, data), axis=0)
        data = src_repeat[:sr]

    return data, sr

