#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import seaborn as sns

from joblib import load
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef,
    precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def plot_cm(cm, labels, model):
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, square=True,
                xticklabels=np.unique(labels), yticklabels=np.unique(labels))

    # Labels, title, and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix {model}')
    ax.xaxis.set_ticklabels(np.unique(labels))
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_ticklabels(np.unique(labels))
    ax.yaxis.set_tick_params(rotation=0)
    plt.show()


def get_test_dataset() -> dict:
    dataset = f'{CURRENT_DIR}/../parsers/datasets/test.csv'
    df = pd.read_csv(dataset)

    X = df.drop('label', axis=1)
    labels = df['label']

    # Create a label encoder object
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    pickle_feat_path = \
        f'{CURRENT_DIR}/../feature_engineering/selected_features_rfc.pickle'

    with open(pickle_feat_path, 'rb') as file:
        # Load the array from the pickle file
        features_selected = pickle.load(file)

    # Use only labels coming from RFE feature selection
    X = X.loc[:, features_selected]

    return (X, y, labels)


def main():
    # Define custom logger
    demo_logger = logging.getLogger('demo_logger')
    demo_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s %(levelname)8s %(process)7d > %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
    )
    demo_logger.addHandler(handler)

    # Loading the model from disk
    svm_model = load(f'{CURRENT_DIR}/../models/svm.joblib')
    knn_model = load(f'{CURRENT_DIR}/../models/knn.joblib')

    X, y, labels = get_test_dataset()

    y_pred_svm = svm_model.predict(X)
    y_pred_knn = knn_model.predict(X)

    # Accuracy
    accuracy_svm = accuracy_score(y, y_pred_svm)
    accuracy_knn = accuracy_score(y, y_pred_knn)
    precision_svm = precision_score(y, y_pred_svm, average='macro')
    precision_knn = precision_score(y, y_pred_knn, average='macro')
    recall_svm = recall_score(y, y_pred_svm, average='macro')
    recall_knn = recall_score(y, y_pred_knn, average='macro')
    f1_svm = f1_score(y, y_pred_svm, average='macro')
    f1_knn = f1_score(y, y_pred_knn, average='macro')
    matthews_svm = matthews_corrcoef(y, y_pred_svm)
    matthews_knn = matthews_corrcoef(y, y_pred_knn)

    demo_logger.info(f"Model accuracy SVM: {accuracy_svm:.3f}")
    demo_logger.info(f"Model accuracy KNN: {accuracy_knn:.3f}")
    demo_logger.info(f'Precision SVM: {precision_svm:0.3f}')
    demo_logger.info(f'Precision KNN: {precision_knn:0.3f}')
    demo_logger.info(f'Recall SVM: {recall_svm:0.3f}')
    demo_logger.info(f'Recall KNN: {recall_knn:0.3f}')
    demo_logger.info(f'F1 SVM: {f1_svm:0.3f}')
    demo_logger.info(f'F1 KNN: {f1_knn:0.3f}')
    demo_logger.info(f'Matthews Correlation Coefficient SVM: {matthews_svm:0.3f}')
    demo_logger.info(f'Matthews Correlation Coefficient KNN: {matthews_knn:0.3f}')

    cm_svm = confusion_matrix(y, y_pred_svm)
    plot_cm(cm_svm, labels, '(SVM)')

    cm_knn = confusion_matrix(y, y_pred_knn)
    plot_cm(cm_knn, labels, '(KNN)')


if __name__ == '__main__':
    main()
