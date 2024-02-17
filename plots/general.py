#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns


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


def scatter_plot(self, cols):
    fig, ax = plt.subplots()
    for label in self.df['label'].unique():
        color = self.get_color(label)
        tmp = self.df[self.df['label'] == label]
        ax.scatter(tmp[cols[0]], tmp[cols[1]], color=color, label=label)
        del tmp

    ax.legend()
    ax.grid(True)

    plt.show()


def label_dist(df):
    sns.countplot(
        x='label',
        data=df,
        order=df['label'].value_counts().index
    )
    plt.title('Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()


def main():
    dataset = f'{CURRENT_DIR}/../parsers/datasets/train.csv'
    df = pd.read_csv(dataset)

    label_dist(df)

    waveplots(df)


if __name__ == '__main__':
    main()
