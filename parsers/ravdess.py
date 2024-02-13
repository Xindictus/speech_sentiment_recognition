#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
import numpy as np
import os
import pandas as pd

from parsers.dataset_parser import DatasetParser


class RavdessParser(DatasetParser):
    EMOTIONS = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
        '08': 'surprised'
    }

    COLS = {
        'file': str,
        'sample_rate': int,
        'modality': int,
        'vocal_channel': int,
        'emotional intensity': int,
        'statement': int,
        'repetition': int,
        'gender': str,
        'mfcc': np.array,
        'mel': np.array,
        'rms': np.array,
        'spce': np.array,
        'zcr': np.array,
        'label': str,
    }

    def get_gender(self, val):
        return 'male' if int(val) % 2 == 1 else 'female'

    def extract_features(self) -> None:
        data = []

        for wav in self.wavList:
            wav_parts = os.path.basename(wav).split('.')[0].split('-')
            y, sr = librosa.load(wav)

            """
            WAV information retrieved from filename in order of `-` splits:
                - modality (wav_parts[0])
                - vocal channel (wav_parts[1])
                - emotion (wav_parts[2])
                - emotional intensity (wav_parts[3])
                - statement (wav_parts[4])
                - repetition (wav_parts[5])
                - male/female actor/actress (wav_parts[6])
            """
            data.append([
                wav,
                sr,
                int(wav_parts[0]),
                int(wav_parts[1]),
                int(wav_parts[3]),
                int(wav_parts[4]),
                int(wav_parts[5]),
                self.get_gender(wav_parts[6]),
                self.mfcc(y, sr),
                self.mel(y, sr),
                self.rms(y),
                self.spce(y, sr),
                self.zcr(y),
                self.EMOTIONS[wav_parts[2]]
            ])

        self.df = pd.DataFrame(data, columns=self.COLS)

        return self
