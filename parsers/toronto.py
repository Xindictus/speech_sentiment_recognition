#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
import numpy as np
import os
import pandas as pd

from parsers.dataset_parser import DatasetParser


class TorontoParser(DatasetParser):
    COLS = {
        'file': str,
        'sample_rate': int,
        'mfcc': np.array,
        'mel': np.array,
        'rms': np.array,
        'stft': np.array,
        'zcr': np.array,
        'label': str,
    }

    def parse(self) -> list:
        files = []
        for root, _, fls in os.walk(self.dataset_path, topdown=False):
            for name in fls:
                files.append(os.path.join(root, name))

        self.wavList = files

        return self

    def extract_features(self) -> None:
        df = pd.DataFrame(
            columns=self.COLS,
        )

        for wav in self.wavList:
            label = os.path.basename(wav).split('_')[2].split('.')[0]
            y, sr = librosa.load(wav)
            df.loc[len(df.index)] = [
                wav,
                sr,
                self.mfcc(y, sr),
                self.mel(y, sr),
                self.rms(y),
                self.stft(y),
                self.zcr(y),
                'pleasant_surprise' if label == 'ps' else label
            ]

        self.df = df

        return self