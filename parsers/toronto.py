#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
import os
import pandas as pd

from parsers.dataset_parser import DatasetParser


class TorontoParser(DatasetParser):
    COLS = {
        'file': str,
        'sample_rate': int
    }

    def label_handling(self, lb) -> str:
        return 'pleasant_surprise' if lb == 'ps' else lb

    def extract_features(self) -> None:
        data = []
        self.finalize_col_labels()

        for wav in self.wavList:
            label = os.path.basename(wav).split('_')[2].split('.')[0]
            y, sr = librosa.load(wav)

            args = (y, sr)
            features = [getattr(self, f)(*args) for f in self.FEATURES]

            data.append([
                wav,
                sr,
                *features,
                self.label_handling(label)
            ])

        self.df = pd.DataFrame(data, columns=self.COLS)

        return self
