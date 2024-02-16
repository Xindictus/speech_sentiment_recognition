import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_dataset() -> dict:
    dataset = f'{CURRENT_DIR}/datasets/train.csv'
    df = pd.read_csv(dataset)

    X = df.drop('label', axis=1)
    y = df['label']

    # Create a label encoder object
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return (X, y)


def get_model_results(X, y) -> dict:
    models = dict()

    for selected_features in range(5, X.shape[1] + 1, 5):
        model = SVC(kernel="linear")
        rfe = RFE(
            estimator=model,
            n_features_to_select=selected_features,
            step=1
        )
        rfe = rfe.fit(X, y)

        models[str(selected_features)] = Pipeline(steps=[
            ('s', rfe),
            ('m', model)
        ])

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

    # feature_selected = rfe.support_

    # # You can also use this mask to transform your dataset
    # X_selected = X.loc[:, feature_selected]

    # print(X_selected.head())


if __name__ == '__main__':
    main()
