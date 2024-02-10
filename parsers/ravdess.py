#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

from parsers.dataset_parser import DatasetParser


class RavdessParser(DatasetParser):
    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

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
        pass
