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

    def label_handling(self, lb) -> str:
        return 'pleasant_surprise' if lb == 'ps' else lb

    def extract_features(self) -> None:
        data = []

        for wav in self.wavList:
            label = os.path.basename(wav).split('_')[2].split('.')[0]
            y, sr = librosa.load(wav)
            data.append([
                wav,
                sr,
                self.mfcc(y, sr),
                self.mel(y, sr),
                self.rms(y),
                self.stft(y),
                self.zcr(y),
                self.label_handling(label)
            ])

        self.df = pd.DataFrame(data, columns=self.COLS)

        return self
