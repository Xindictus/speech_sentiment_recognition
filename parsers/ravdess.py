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
        'actor': str,
        'type': str
    }

    def src(self) -> str:
        return 'ravdess'

    def get_gender(self, val):
        return 'M' if int(val) % 2 == 1 else 'F'

    def post_process_extras(self):
        self.features['actor'] = self.df['actor']
        self.features['gender'] = self.df['gender']
        self.features['type'] = self.df['type']
        return self

    def extract_features(self) -> None:
        data = []
        self.finalize_col_labels()

        for wav in self.wavList:
            wav_parts = os.path.basename(wav).split('.')[0].split('-')
            y, sr = librosa.load(wav)

            args = (y, sr)
            features = [getattr(self, f)(*args) for f in self.FEATURES]

            y_noise = self.noise(y)
            args_noise = (y_noise, sr)
            features_noise = [getattr(self, f)(*args_noise) for f in self.FEATURES]

            y_stretch = self.stretch(y)
            args_stretch = (y_stretch, sr)
            features_stretch = [getattr(self, f)(*args_stretch) for f in self.FEATURES]

            # y_pitch_shift = self.pitch_shift(y, sr)
            # args_pitch_shift = (y_pitch_shift, sr)
            # features_pitch_shift = [getattr(self, f)(*args_pitch_shift) for f in self.FEATURES]

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
            data.extend([[
                wav,
                sr,
                int(wav_parts[0]),
                int(wav_parts[1]),
                int(wav_parts[3]),
                int(wav_parts[4]),
                int(wav_parts[5]),
                self.get_gender(wav_parts[6]),
                f'{int(wav_parts[6])}',
                'original',
                *features,
                self.EMOTIONS[wav_parts[2]]
            ], [
                wav,
                sr,
                int(wav_parts[0]),
                int(wav_parts[1]),
                int(wav_parts[3]),
                int(wav_parts[4]),
                int(wav_parts[5]),
                self.get_gender(wav_parts[6]),
                f'{int(wav_parts[6])}',
                'noise',
                *features_noise,
                self.EMOTIONS[wav_parts[2]]
            ], [
                wav,
                sr,
                int(wav_parts[0]),
                int(wav_parts[1]),
                int(wav_parts[3]),
                int(wav_parts[4]),
                int(wav_parts[5]),
                self.get_gender(wav_parts[6]),
                f'{int(wav_parts[6])}',
                'stretch',
                *features_stretch,
                self.EMOTIONS[wav_parts[2]]
            ]])
            # , [
            #     wav,
            #     sr,
            #     int(wav_parts[0]),
            #     int(wav_parts[1]),
            #     int(wav_parts[3]),
            #     int(wav_parts[4]),
            #     int(wav_parts[5]),
            #     self.get_gender(wav_parts[6]),
            #     f'{int(wav_parts[6])}',
            #     *features_pitch_shift,
            #     self.EMOTIONS[wav_parts[2]]
            # ]

        self.df = pd.DataFrame(data, columns=self.COLS)

        return self
