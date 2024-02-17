#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import time

from joblib import dump
from sklearn.metrics import (
    accuracy_score, auc,
    confusion_matrix, f1_score,
    matthews_corrcoef, precision_score,
    recall_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize, LabelEncoder

# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_dataset() -> dict:
    dataset = f'{CURRENT_DIR}/../parsers/datasets/train.csv'
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


def plot_cm(cm, y_test, labels):
    # Create a heatmap
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, square=True,
                xticklabels=np.unique(labels), yticklabels=np.unique(labels))

    # Labels, title, and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(np.unique(labels))
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_ticklabels(np.unique(labels))
    ax.yaxis.set_tick_params(rotation=0)
    plt.show()


def plot_roc(y, y_test, y_score, labels):
    n_classes = len(np.unique(y))
    y_test = label_binarize(y_test, classes=np.unique(y))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Colors for the ROC curves
    # Choose a colormap
    colormap = plt.get_cmap('tab20')

    # Generate colors from the colormap
    colors = [colormap(i / n_classes) for i in range(n_classes)]

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label='ROC curve of class {0} (area = {1:0.2f})'.format(
                labels[i], roc_auc[i]
            )
        )
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC for {0} Classes'.format(n_classes))
    plt.legend(loc="lower right")
    plt.show()


def main():
    # Define custom logger
    knn_logger = logging.getLogger('knn_logger')
    knn_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            '%(asctime)s %(levelname)8s %(process)7d > %(message)s',
            '%Y-%m-%d %H:%M:%S'
        )
    )
    knn_logger.addHandler(handler)

    X, y, labels = get_dataset()

    # breakdown to train/validation
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    start = time.time()

    knn = KNeighborsClassifier(
        n_neighbors=11,
        p=7,
        metric='euclidean'
    )

    knn.fit(X_train, y_train)

    end = time.time()
    knn_logger.info(f'KNN model trained in: {(end - start):0.2f}s')

    y_score = knn.predict_proba(X_test)
    y_pred = knn.predict(X_test)

    # Save knn trained model
    dump(knn, f'{CURRENT_DIR}/knn.joblib')

    # METRICS
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    matthews = matthews_corrcoef(y_test, y_pred)

    print(f'Accuracy: {accuracy:0.3f}')
    print(f'precision: {precision:0.3f}')
    print(f'recall: {recall:0.3f}')
    print(f'f1: {f1:0.3f}')
    print(f'Matthews Correlation Coefficient: {matthews:0.3f}')

    cm = confusion_matrix(y_test, y_pred)
    plot_cm(cm, y_test, labels)

    plot_roc(y, y_test, y_score, np.unique(labels))


if __name__ == '__main__':
    main()
