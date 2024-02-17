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


def scatter_plot(df):

    for i in range(len(df.columns) - 2):
        fig, ax = plt.subplots()

        curr_col = df.columns[i]
        next_col = df.columns[i + 1]

        # Accessing the data for the current and next column
        current_data = df[curr_col]
        next_data = df[next_col]

        print(f"Pair: ({curr_col}, {next_col})")

        for label in df['label'].unique():
            color = get_color(label)
            tmp = df[df['label'] == label]
            ax.scatter(tmp[curr_col], tmp[next_col], color=color, label=label)
            del tmp

        ax.set_xlabel(curr_col)
        ax.set_ylabel(next_col)
        ax.legend()
        ax.grid(True)

        plt.savefig(f'{CURRENT_DIR}/scatters/{curr_col}-{next_col}.png')


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

    scatter_plot(df)


if __name__ == '__main__':
    main()
