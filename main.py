#!/usr/bin/env python

import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import IPython.display

# Make it look nice
seaborn.set(style='ticks')

# import the sound you wanna fuck with
audio_path = 'raigeki.mp3'

y, sr = librosa.load(audio_path)
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_S = librosa.logamplitude(S, ref_power=np.max)

# Display the things and make it pretty
plt.figure(figsize=(20, 20))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title=('mel spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.savefig('swag.png')
