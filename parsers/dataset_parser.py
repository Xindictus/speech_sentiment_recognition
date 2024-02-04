#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from os.path import exists


class DatasetParser(ABC):
    def __init__(self, _dataset_path):
        if self.is_valid_path(_dataset_path):
            self.dataset_path = _dataset_path
            self.wavList = []
        else:
            raise ValueError('path.not_exists')

    def is_valid_path(self, path: str) -> bool:
        """_summary_

        Args:
            path (str): Absolute path to file/folder

        Returns:
            bool: Informs of path validity
        """
        return exists(path)

    @abstractmethod
    def parse(self) -> None:
        """_summary_
        This function will include the logic of parsing
        the Kaggle audio files and creating a dataset
        out of them
        """
        raise NotImplementedError()

    def test(self):
        fig = plt.figure(figsize=(15, 15))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        for i, label in enumerate(self.df['label'].unique()):
            fn = self.df.loc[self.df['label'] == label]
            fig.add_subplot(5, 2, i + 1)
            plt.title(label)
            data, sample_rate = librosa.load(fn)
            librosa.display.waveplot(data, sr=sample_rate)
        plt.savefig('class_examples.png')
