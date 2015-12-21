#!/usr/bin/env python

import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import seaborn
import IPython.display

# Make it look nice
seaborn.set(style='ticks')

# import the sound you wanna fuck with
audio_path = 'luka.mp3'

y, sr = librosa.load(audio_path)

# Number of times to run the Mel Filter on the source
S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

# dat log shit doe
log_S = librosa.logamplitude(S, ref_power=np.max)

# Display the things and make it pretty
plt.figure(figsize=(40, 20))
# cscheme = cm.get_cmap('PuOr')
# cscheme = cm.get_cmap('coolwarm')

# cscheme.set_under('k')

librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

plt.title=('mel spectrogram')
plt.xlabel('Time')
plt.ylabel('Hz')
plt.colorbar(format='%+02.0f dB')
plt.savefig('ginseng mel.png')

y_harmonic, y_percussive = librosa.effects.hpss(y)

# What do the spectrograms look like?
# Let's make and display a mel-scaled power (energy-squared) spectrogram
S_harmonic   = librosa.feature.melspectrogram(y_harmonic, sr=sr)
S_percussive = librosa.feature.melspectrogram(y_percussive, sr=sr)

# Convert to log scale (dB). We'll use the peak power as reference.
log_Sh = librosa.logamplitude(S_harmonic, ref_power=np.max)
log_Sp = librosa.logamplitude(S_percussive, ref_power=np.max)

# Make a new figure
plt.figure(figsize=(12,6))

plt.subplot(2,1,1)
# Display the spectrogram on a mel scale
librosa.display.specshow(log_Sh, sr=sr, y_axis='mel')

# Put a descriptive title on the plot
plt.title=('mel power spectrogram (Harmonic)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

plt.subplot(2,1,2)
librosa.display.specshow(log_Sp, sr=sr, x_axis='time', y_axis='mel')

# Put a descriptive title on the plot
plt.title=('mel power spectrogram (Percussive)')

# draw a color bar
plt.colorbar(format='%+02.0f dB')

# Make the figure layout compact
plt.tight_layout()

plt.savefig('ginseng seperates.png')

# We'll use a CQT-based chromagram here.  An STFT-based implementation also
# exists in chroma_cqt()
# We'll use the harmonic component to avoid pollution from transients
C = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# Make a new figure
plt.figure(figsize=(12,4))

# Display the chromagram: the energy in each chromatic pitch class as a function
# of time
# To make sure that the colors span the full range of chroma values, set vmin
# and vmax
librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0,
        vmax=1)

plt.title=('Chromagram')
plt.colorbar()

plt.tight_layout()

plt.savefig('ginseng chroma.png')
