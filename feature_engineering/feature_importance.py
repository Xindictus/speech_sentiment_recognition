#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_color(lbl):
    return {
        'fear': 'red',
        'disgust': 'blue',
        'pleasant_surprise': 'green',
        'sad': 'yellow',
        'angry': 'purple',
        'neutral': 'orange',
        'happy': 'pink'
    }[lbl]


def scatter_plot(df, f1, f2):
    print(f1)
    print(f2)
    fig, ax = plt.subplots()
    for label in df['label'].unique():
        color = get_color(label)
        tmp = df[df['label'] == label]
        ax.scatter(tmp[f1], tmp[f2], color=color, label=label)
        del tmp

    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    ax.legend()
    ax.grid(True)

    plt.show()


def main():
    dataset = f'{CURRENT_DIR}/../parsers/datasets/train.csv'
    df = pd.read_csv(dataset)

    # feature matrix
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    # Model training
    start = time.time()

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    end = time.time()
    print(f"Total runtime for model training: {(end - start):0.2f}s")

    # Accuracy
    y_pred = clf.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))

    # Compute importances
    start = time.time()

    importances = clf.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)

    end = time.time()
    print(f"Elapsed time to compute the importances: {(end - start):0.2f}s")

    # Create a DataFrame for visualization
    features = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print(features)
    scatter_plot(df, features.iat[0, 0], features.iat[1, 0])

    plt.figure(figsize=(10, 6))
    plt.barh(features['Feature'], features['Importance'])
    plt.xlabel('Importance')
    plt.yticks(visible=False)
    plt.title('Feature Importance')
    # Invert y-axis to have the most important feature on top
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == '__main__':
    main()
