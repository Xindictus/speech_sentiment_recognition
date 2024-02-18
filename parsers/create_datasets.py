#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import pandas as pd
import time

from parsers.ravdess import RavdessParser
from parsers.toronto import TorontoParser

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Required for merging the datasets
SPLIT_RATIO = 0.1


def sample_sentiments(df, ratio=None):
    frac = ratio if ratio is not None else SPLIT_RATIO
    return df.groupby(['label', 'type']).sample(
        frac=frac,
        random_state=42
    )


def main():
    # Define custom logger
    parser_logger = logging.getLogger('parser_logger')
    parser_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s %(levelname)8s %(process)7d > %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
    )
    parser_logger.addHandler(handler)

    ravdess_path = f'{CURRENT_DIR}/datasets/ravdess'
    toronto_path = f'{CURRENT_DIR}/datasets/toronto'

    ravdess = RavdessParser(ravdess_path)
    toronto = TorontoParser(toronto_path)

    # Toronto extraction
    start = time.time()

    toronto_export = f'{CURRENT_DIR}/datasets/toronto.csv'
    toronto_features = f'{CURRENT_DIR}/datasets/toronto_features.csv'
    toronto_dist = f'{CURRENT_DIR}/../plots/toronto_waveplots.png'
    toronto_spec = f'{CURRENT_DIR}/../plots/toronto_spectrograms.png'
    toronto \
        .parse() \
        .extract_features() \
        .waveplots(toronto_dist) \
        .spectrograms(toronto_spec) \
        .post_processing() \
        .post_process_extras() \
        .export(toronto_export, 'dataset') \
        .export(toronto_features)

    del toronto

    end = time.time()
    parser_logger.info(f'Total runtime (toronto): {(end - start):0.2f}s')

    # Ravdess extraction
    start = time.time()

    ravdess_export = f'{CURRENT_DIR}/datasets/ravdess.csv'
    ravdess_features = f'{CURRENT_DIR}/datasets/ravdess_features.csv'
    ravdess_dist = f'{CURRENT_DIR}/../plots/ravdess_waveplots.png'
    ravdess_spec = f'{CURRENT_DIR}/../plots/ravdess_spectrograms.png'
    ravdess \
        .parse() \
        .extract_features() \
        .waveplots(ravdess_dist) \
        .spectrograms(ravdess_spec) \
        .post_processing() \
        .post_process_extras() \
        .export(ravdess_export, 'dataset') \
        .export(ravdess_features)

    del ravdess

    end = time.time()
    parser_logger.info(f'Total runtime (ravdess): {(end - start):0.2f}s')

    # Dataset merge and split to train/test
    parser_logger.info('Merging sentiments')

    toronto_df = pd.read_csv(toronto_features)
    ravdess_df = pd.read_csv(ravdess_features)

    # Keeping similar sentiments
    sentiments = [
        'angry',
        'fear',
        'happy',
        'neutral',
        'sad',
    ]

    toronto_df = toronto_df[toronto_df['label'].isin(sentiments)]
    ravdess_df = ravdess_df[ravdess_df['label'].isin(sentiments)]

    parser_logger.info('Spliting to train/test set')

    # Sample datasets and create training set
    toronto_sampled = toronto_df[toronto_df['age'] == 'Y']
    # keep ratios per actor
    toronto_sampled = sample_sentiments(toronto_sampled, 0.1)
    toronto_sampled = toronto_sampled.drop(columns=[
        'age',
        'type'
    ])

    ravdess_sampled = ravdess_df[ravdess_df['actor'] <= 20]
    ravdess_sampled = ravdess_sampled.drop(columns=[
        'actor',
        'gender',
        'type'
    ])

    training_df = pd.concat([toronto_sampled, ravdess_sampled])
    train_csv_path = f'{CURRENT_DIR}/datasets/train.csv'

    # Create test set with remaining original
    toronto_rest = toronto_df[(toronto_df['age'] == 'O') &
                              (toronto_df['type'] == 'original')]
    # keep ratios per actor
    toronto_rest = sample_sentiments(toronto_rest, 0.03)
    toronto_rest = toronto_rest.drop(columns=[
        'age',
        'type'
    ])

    ravdess_rest = ravdess_df[(ravdess_df['actor'] > 20) &
                              (ravdess_df['gender'] == 'F') &
                              (ravdess_df['type'] == 'original')]
    ravdess_rest = ravdess_rest.drop(columns=[
        'actor',
        'gender',
        'type'
    ])

    test_df = pd.concat([toronto_rest, ravdess_rest])
    test_csv_path = f'{CURRENT_DIR}/datasets/test.csv'

    # Export training/test datasets
    training_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)


if __name__ == '__main__':
    main()
