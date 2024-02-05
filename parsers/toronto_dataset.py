#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os

from toronto import TorontoParser


# Define custom logger
parser_logger = logging.getLogger('api_logger')

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    dataset_path = f'{CURRENT_DIR}/datasets/toronto'
    parser = TorontoParser(dataset_path)

    parser \
        .parse() \
        .extract_features() \
        .export('feat.csv')


if __name__ == '__main__':
    main()
