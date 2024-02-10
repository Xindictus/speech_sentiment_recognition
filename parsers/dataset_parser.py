#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from abc import ABC, abstractmethod
from os.path import exists


class DatasetParser(ABC):
    FEATURES = [
        'mfcc',
        'mel',
        'rms',
        'spce',
        'stft',
        'zcr'
    ]

    def __init__(self, _dataset_path):
        if self._is_valid_path(_dataset_path):
            self.dataset_path = _dataset_path
            self.wavList = []
        else:
            raise ValueError('path.not_exists')

    def parse(self) -> None:
        files = []
        for root, _, fls in os.walk(self.dataset_path, topdown=False):
            for name in fls:
                files.append(os.path.join(root, name))

        self.wavList = files

        return self

    @abstractmethod
    def extract_features(self) -> None:
        """_summary_
        This function will include the logic of parsing
        the Kaggle audio files and creating a dataset
        out of them
        """
        raise NotImplementedError()

    def _is_valid_path(self, path: str) -> bool:
        """_summary_

        Args:
            path (str): Absolute path to file/folder

        Returns:
            bool: Informs of path validity
        """
        return exists(path)

    def noise(y):
        noise_amp = 0.035 * np.random.uniform() * np.amax(y)
        return (y + noise_amp*np.random.normal(size=y.shape[0]))

    ######################################
    #              Features              #
    ######################################
    def chroma_stft(self, y, sr) -> np.array:
        """_summary_

        Compute a chromagram from a waveform or power spectrogram.

        Args:
            y: audio time series
            sr: sample rate

        Returns:
            np.array
        """
        # Chroma_stft
        stft = np.abs(librosa.stft(y))
        return np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    def mfcc(self, y, sr) -> np.array:
        """_summary_
        Mel-frequency cepstral coefficients (MFCCs)

        Args:
            y: audio time series
            sr: sample rate

        Returns:
            np.array
        """
        return np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)

    def mel(self, y, sr) -> np.array:
        """_summary_

        Compute a mel-scaled spectrogram.

        Args:
            y: audio time series
            sr: sample rate

        Returns:
            np.array
        """
        return np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)

    def rms(self, y) -> np.array:
        """_summary_

        Compute root-mean-square (RMS) value for each frame,
        either from the audio samples y or from a spectrogram S.

        Args:
            y: audio time series

        Returns:
            np.array
        """
        return np.mean(librosa.feature.rms(y=y).T, axis=0)

    def spce(self, y, sr) -> np.array:
        """_summary_

        Compute the spectral centroid.

        Each frame of a magnitude spectrogram is normalized and
        treated as a distribution over frequency bins, from which
        the mean (centroid) is extracted per frame.

        Args:
            y: audio time series
            sr: sample rate

        Returns:
            np.array
        """
        return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    def stft(self, y) -> np.array:
        """_summary_
        Short-time Fourier transform (STFT).

        The STFT represents a signal in the time-frequency domain by
        computing discrete Fourier transforms (DFT) over short
        overlapping windows.

        Args:
            y: audio time series

        Returns:
            np.array
        """
        return np.mean(librosa.stft(y=y).T, axis=0)

    def zcr(self, y) -> np.array:
        """_summary_
        Compute the zero-crossing rate of an audio time series.

        Args:
            y: audio time series

        Returns:
            np.array
        """
        return np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

    ######################################
    #     Post-processing of features    #
    ######################################
    def padding(self):
        # Pad features and add to feature array
        for col in self.COLS:
            if col in self.FEATURES:
                max = self.df[col].apply(lambda x: len(x)).max()
                self.df[col] = self.df[col] \
                    .apply(lambda x: np.pad(x, (0, max - len(x))))

    def post_processing(self):
        self.padding()

        # Stack features
        self.df['feature_arr'] = pd.Series(
            [np.array([]) for _ in range(len(self.df))]
        )
        self.df['feature_arr'] = self.df.apply(
            lambda row: np.hstack([row[i] for i in self.FEATURES]),
            axis=1
        )

        feat_df = pd.DataFrame(
            self.df['feature_arr'].tolist(),
            index=self.df.index
        )
        feat_df.columns = [f'feature_{i + 1}' for i in range(feat_df.shape[1])]
        feat_df['label'] = self.df['label']
        self.df = feat_df

        return self

    ######################################
    #              Plotting              #
    ######################################
    def waveplots(self):
        fig = plt.figure(figsize=(15, 15))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i, label in enumerate(self.df['label'].unique()):
            fn = self.df.loc[self.df['label'] == label]
            fig.add_subplot(5, 2, i + 1)
            plt.title(label)
            data, sample_rate = librosa.load(fn)
            librosa.display.waveplot(data, sr=sample_rate)
        plt.savefig('class_examples.png')

    def get_color(self, lbl):
        return {
            'fear': 'red',
            'disgust': 'blue',
            'pleasant_surprise': 'green',
            'sad': 'yellow',
            'angry': 'purple',
            'neutral': 'orange',
            'happy': 'pink'
        }[lbl]

    def scatter_plot(self, cols):
        fig, ax = plt.subplots()
        for label in self.df['label'].unique():
            color = self.get_color(label)
            tmp = self.df[self.df['label'] == label]
            ax.scatter(tmp[cols[0]], tmp[cols[1]], color=color, label=label)
            del tmp

        ax.legend()
        ax.grid(True)

        plt.show()

    def label_dist(self):
        sns.countplot(
            x='label',
            data=self.df,
            order=self.df['label'].value_counts().index
        )
        plt.title('Count of Each Label')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.show()

    ######################################
    #             DF Handling            #
    ######################################
    def array2string(self, npArr):
        array_settings = {
            'data': npArr.tolist(),
            'shape': npArr.shape,
            'dtype': str(npArr.dtype)
        }

        return json.dumps(array_settings)

    def string2array(self, str):
        array_settings = json.loads(str)
        arr = np.array(array_settings['data'], dtype=array_settings['dtype'])
        return arr.reshape(array_settings['shape'])

    def export(self, path):
        # exp_df = self.df.copy()
        # exp_df['mfcc'] = exp_df['mfcc'].apply(lambda x: self.array2string(x))
        # exp_df['zcr'] = exp_df['zcr'] \
        #     .apply(lambda x: self.array2string(x))
        # exp_df['stft'] = exp_df['stft'].apply(lambda x: self.array2string(x))
        self.df.to_csv(path, index=False)

    def import_df(self):
        self.df = pd.read_csv(self.dataset_path)
        # self.df['mfcc'] = self.df['mfcc'].apply(lambda x: self.string2array(x))
        # self.df['zcr'] = self.df['zcr'] \
        #     .apply(lambda x: self.string2array(x))

        # self.df['mfcc'] = self.df['mfcc'].apply(lambda x: np.std(x, axis=0))
        # self.df['zero_crossing_rate'] = self.df['zero_crossing_rate'] \
        #     .apply(lambda x: np.std(x, axis=0))
        # self.df['mfcc'] = self.df['mfcc'].apply(np.mean)
        # self.df['zero_crossing_rate'] = self.df['zero_crossing_rate'] \
        #     .apply(np.mean)
