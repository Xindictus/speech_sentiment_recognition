import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn

import glob

import librosa
import librosa.display
import IPython.display as ipd

data_path = (r'C:\Users\Mihalis\Desktop\ML\archive\*\*.wav')
data_ravdess_paths = glob.glob(data_path)

y, sr = librosa.load(data_ravdess_paths[0])
stft = np.abs(librosa.stft(y))
chroma = librosa.feature.chroma_stft(y=y, sr=sr,)

#Freq waveshow
fig, ax = plt.subplots()
librosa.display.waveshow(y=stft, sr=sr, ax=ax)
ax.set(title='stft waveplot')
plt.show()

#Freq waveshow
fig, ax = plt.subplots()
librosa.display.waveshow(y, sr=sr, ax=ax)
ax.set(title='Instance Waveplot')

#Chroma wa
fig, ax = plt.subplots()
librosa.display.waveshow(chroma, sr=sr, ax=ax)
ax.set(title='Chroma Waveplot')
plt.show()

#Chroma specshow
fig, ax = plt.subplots()
img = librosa.display.specshow(chroma, sr=sr, y_axis='chroma', x_axis='time')
fig.colorbar(img, ax=ax)
ax.set(title = 'Chroma Spectogram')

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(chroma, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='chroma', sr=sr, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Chroma to db Spectogram')
plt.show()

#Mel spectogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')


#Stft
d = librosa.stft(y)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(np.abs(d), ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='ms', y_axis='chroma', sr=sr, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Stft to db spectrogram')
plt.show()
