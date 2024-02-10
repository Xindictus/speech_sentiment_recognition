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
