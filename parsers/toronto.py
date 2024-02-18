#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
import os
import pandas as pd

from parsers.dataset_parser import DatasetParser


class TorontoParser(DatasetParser):
    COLS = {
        'file': str,
        'sample_rate': int,
        'age': str,
        'type': str
    }

    def src(self) -> str:
        return 'toronto'

    def label_handling(self, lb) -> str:
        return 'pleasant_surprise' if lb == 'ps' else lb

    def post_process_extras(self):
        self.features['age'] = self.df['age']
        self.features['type'] = self.df['type']
        return self

    def extract_features(self) -> None:
        data = []
        self.finalize_col_labels()

        for wav in self.wavList:
            label = os.path.basename(wav).split('_')[2].split('.')[0]
            y, sr = librosa.load(wav)

            args = (y, sr)
            features = [getattr(self, f)(*args) for f in self.FEATURES]

            y_noise = self.noise(y)
            args_noise = (y_noise, sr)
            features_noise = [getattr(self, f)(*args_noise) for f in self.FEATURES]

            y_stretch = self.stretch(y)
            args_stretch = (y_stretch, sr)
            features_stretch = [getattr(self, f)(*args_stretch) for f in self.FEATURES]

            data.extend([[
                wav,
                sr,
                os.path.basename(wav)[0],
                'original',
                *features,
                self.label_handling(label),
            ], [
                wav,
                sr,
                os.path.basename(wav)[0],
                'noise',
                *features_noise,
                self.label_handling(label)
            ], [
                wav,
                sr,
                os.path.basename(wav)[0],
                'stretch',
                *features_stretch,
                self.label_handling(label)
            ]])

        self.df = pd.DataFrame(data, columns=self.COLS)

        return self
