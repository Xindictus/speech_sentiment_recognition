#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
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
        'actor': str
    }

    def src(self) -> str:
        return 'ravdess'

    def get_gender(self, val):
        return 'male' if int(val) % 2 == 1 else 'female'

    def post_process_extras(self):
        self.features['actor'] = self.df['actor']
        return self

    def extract_features(self) -> None:
        data = []
        self.finalize_col_labels()

        for wav in self.wavList:
            wav_parts = os.path.basename(wav).split('.')[0].split('-')
            y, sr = librosa.load(wav)

            args = (y, sr)
            features = [getattr(self, f)(*args) for f in self.FEATURES]

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
                f'{int(wav_parts[6])}',
                *features,
                self.EMOTIONS[wav_parts[2]]
            ])

        self.df = pd.DataFrame(data, columns=self.COLS)

        return self
