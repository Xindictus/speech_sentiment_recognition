#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import os

# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d
import numpy as np

from sklearn import decomposition


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    dataset = f'{CURRENT_DIR}/../parsers/datasets/train.csv'
    df = pd.read_csv(dataset)

    # feature matrix
    X = df.drop('label', axis=1)
    y = df['label']

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()

    ax = fig.add_subplot(111, projection="3d", elev=48, azim=134)
    ax.set_position([0, 0, 0.95, 1])

    plt.cla()
    pca = decomposition.PCA(n_components=3)
    pca.fit(X)
    X = pca.transform(X)

    # Combine into a DataFrame for ease of use
    pca_df = pd.DataFrame({
        'PC1': X[:, 0],
        'PC2': X[:, 1],
        'PC3': X[:, 2],
        'Label': y
    })

    colors = {
        'fear': 'r',
        'disgust': 'g',
        'pleasant_surprise': 'k',
        'sad': 'b',
        'angry': 'y',
        'neutral': 'c',
        'happy': 'm'
    }

    for label in np.unique(y):
        ix = df['label'] == label
        ax.scatter(
            pca_df.loc[ix, 'PC1'],
            pca_df.loc[ix, 'PC2'],
            pca_df.loc[ix, 'PC3'],
            c=colors[label],
            label=label,
            cmap=plt.cm.nipy_spectral,
            edgecolor="k",
            s=50
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.show()


if __name__ == '__main__':
    main()
