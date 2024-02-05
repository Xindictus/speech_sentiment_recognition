#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import librosa
import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from os.path import exists


class DatasetParser(ABC):
    def __init__(self, _dataset_path, _input=None):
        if self.is_valid_path(_dataset_path):
            self.dataset_path = _dataset_path
            self.wavList = []
            self.df = _input
        else:
            raise ValueError('path.not_exists')

    def is_valid_path(self, path: str) -> bool:
        """

        Args:
            path (str): Absolute path to file/folder

        Returns:
            bool: Informs of path validity
        """
        return exists(path)

    @abstractmethod
    def parse(self) -> None:
        """
        This function will include the logic of parsing
        the Kaggle audio files and creating a dataset
        out of them
        """
        raise NotImplementedError()

    def mfcc(self, y, sr) -> np.array:
        """
        Mel-frequency cepstral coefficients (MFCCs)

        Returns:
            np.array
        """
        return np.mean(librosa.feature.mfcc(y=y, sr=sr).T, axis=0)

    def zero_crossing_rate(self, y) -> np.array:
        """
        Compute the zero-crossing rate of an audio time series.

        Returns:
            np.array
        """
        return np.mean(librosa.feature.zero_crossing_rate(y=y).T, axis=0)

    def fourier_transforms(self, y) -> np.array:
        """
        Short-time Fourier transform (STFT).

        The STFT represents a signal in the time-frequency domain by
        computing discrete Fourier transforms (DFT) over short overlapping windows.

        Returns:
            np.array
        """
        return np.mean(librosa.stft(y=y).T, axis=0)

    def post_processing(self):
        # Pad MFCCs
        mfcc_max = self.df['mfcc'].apply(lambda x: len(x)).max()
        self.df['mfcc'] = self.df['mfcc'] \
            .apply(lambda x: np.pad(x, (0, mfcc_max - len(x))))

        # Pad STFTs
        mfcc_max = self.df['stft'].apply(lambda x: len(x)).max()
        self.df['stft'] = self.df['stft'] \
            .apply(lambda x: np.pad(x, (0, mfcc_max - len(x))))

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

    def export(self, nm):
        exp_df = self.df.copy()
        exp_df['mfcc'] = exp_df['mfcc'].apply(lambda x: self.array2string(x))
        exp_df.to_csv(nm, index=False)
