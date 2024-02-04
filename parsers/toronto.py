#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd

from .dataset_parser import DatasetParser


class TorontoParser(DatasetParser):
    COLS = {
        'file': str,
        'xxxxxx': str,
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
            df.loc[len(df.index)] = [
                wav,
                '123',
                'pleasant_surprise' if label == 'ps' else label
            ]

        self.df = df

        return self
