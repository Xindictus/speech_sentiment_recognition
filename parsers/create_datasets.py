#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import time

from parsers.ravdess import RavdessParser
from parsers.toronto import TorontoParser

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


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

    toronto \
        .parse() \
        .extract_features() \
        .post_processing() \
        .export('toronto.csv', 'dataset') \
        .export('toronto_features.csv')

    end = time.time()
    parser_logger.info(f"Total process runtime (toronto): {(end - start):0.2f}s")

    # Ravdess extraction
    start = time.time()

    ravdess \
        .parse() \
        .extract_features() \
        .post_processing() \
        .export('ravdess.csv', 'dataset') \
        .export('ravdess_features.csv')

    end = time.time()
    parser_logger.info(f"Total process runtime (ravdess): {(end - start):0.2f}s")


if __name__ == '__main__':
    main()
