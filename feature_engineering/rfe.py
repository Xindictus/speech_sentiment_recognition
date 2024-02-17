#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_dataset() -> dict:
    dataset = f'{CURRENT_DIR}/../parsers/datasets/train.csv'
    df = pd.read_csv(dataset)

    X = df.drop('label', axis=1)
    y = df['label']

    # Create a label encoder object
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return (X, y)


def get_model_results(X, y) -> dict:
    models = dict()

    for n_features_to_select in range(5, X.shape[1] + 1, 5):
        # if n_features_to_select != 125:
        #     continue
        model = SVC(kernel="linear")
        rfe = RFE(
            estimator=model,
            n_features_to_select=n_features_to_select,
            step=1
        )
        rfe = rfe.fit(X, y)

        models[str(n_features_to_select)] = Pipeline(steps=[
            ('s', rfe),
            ('m', model)
        ])

        # We have chosen 125 as the best number of features
        # to train with
        if n_features_to_select == 125:
            features_selected = rfe.support_

            labels = [lab for lab, selected in zip(
                X.columns,
                features_selected) if selected]

            with open(f'{CURRENT_DIR}/selected_features.pickle', 'wb') as file:
                pickle.dump(labels, file)

    return models


def model_evaluation(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

    # evaluate model
    n_scores = cross_val_score(
        model,
        X,
        y,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        error_score='raise'
    )

    return n_scores


def main():
    X, y = get_dataset()
    models = get_model_results(X, y)

    res = list()
    nms = list()

    for nm, md in models.items():
        score = model_evaluation(md, X, y)
        res.append(score)
        nms.append(nm)
        print('Accuracy with %s features: %.3f (%.3f)'
              % (nm, np.mean(score), np.std(score)))

    plt.boxplot(res, labels=nms, showmeans=True)
    plt.show()


if __name__ == '__main__':
    main()
