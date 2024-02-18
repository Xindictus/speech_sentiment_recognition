import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Assuming X_train, y_train are your training data and labels
# Define PY script folder
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_dataset():
    dataset = f'{CURRENT_DIR}/../parsers/datasets/train.csv'
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
    X, y, _ = get_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Create a pipeline that standardizes data then applies SVM
    pipeline = make_pipeline(SVC(kernel='rbf'))

    # Define parameter grid
    param_grid = {
        'svc__C': np.arange(1, 3.1, 0.5),
        'svc__gamma': np.arange(0.007, 0.011, 0.001)
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)


if __name__ == '__main__':
    main()
