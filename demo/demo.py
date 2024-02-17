#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
import pickle

from joblib import load
from sklearn.metrics import (
    accuracy_score, auc,
    confusion_matrix, f1_score,
    matthews_corrcoef, precision_score,
    recall_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_test_dataset() -> dict:
    dataset = f'{CURRENT_DIR}/../parsers/datasets/test.csv'
    df = pd.read_csv(dataset)

X = df.drop('label', axis=1)
    labels = df['label']

    # Create a label encoder object
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    pickle_feat_path = \
        f'{CURRENT_DIR}/../feature_engineering/selected_features.pickle'

    with open(pickle_feat_path, 'rb') as file:
        # Load the array from the pickle file
        features_selected = pickle.load(file)

    # Use only labels coming from RFE feature selection
    X = X.loc[:, features_selected]

    return (X, y, labels)


def main():
    # Loading the model from disk
    model = load(f'{CURRENT_DIR}/../models/svm.joblib')

    X, y, _ = get_test_dataset()
    y_pred = model.predict(X)

    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Model accuracy: {accuracy:.3f}")


if __name__ == '__main__':
    main()
