#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import time

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

    dataset_path = f'{CURRENT_DIR}/datasets/toronto'
    parser = TorontoParser(dataset_path)

    start = time.time()
    parser \
        .parse() \
        .extract_features() \
        .post_processing() \
        .export('feat.csv')

    end = time.time()
    parser_logger.info(f"Total process runtime: {(end - start):0.2f}s")


if __name__ == '__main__':
    main()
