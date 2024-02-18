# ML Project - Speech Sentiment Recognition

Sentiment Recognition in speech using Machine Learning (ML)

<!-- https://badgen.net/badge/:subject/:status/:color?icon=github -->
![python version](https://badgen.net/badge/python/3.11/blue)
![pre-commit](https://badgen.net/badge/pre-commit/3.6.0/green)

- [ML Project - Speech Sentiment Recognition](#ml-project---speech-sentiment-recognition)
  - [1. Description](#1-description)
  - [2. Setup of the environment](#2-setup-of-the-environment)
  - [3. Script execution](#3-script-execution)

## 1. Description

- We explore sentiment analysis in womens' speech for the following sentiments:
  - anger
  - fear
  - happiness
  - sadness
  - neutral

- We extract features from the audio files with the help of `librosa`:
  - `mfcc`
  - `mel`
  - `rms`
  - `spce`
  - `zcr`
  - `chroma` features
  - `pitch`

  > Not all of them were used in the final training.

- The classifiers tuned and trained were:
  - `SVM`
  - `KNN`

- The datasets used are:
  - [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
  - [Ravdess](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

## 2. Setup of the environment

1. [Install pip](https://pip.pypa.io/en/stable/installing/).
2. [Create a virtual enviroment](https://docs.python.org/3/tutorial/venv.html) .
3. Install project depedencies:

   a. Using `pip`:
   - Go into the app directory.
   - Create a new virtual environment:
     > `python3 -m venv env`
   - Source your virtual env:
     > `source env/bin/activate`
   - Install requirements with:
     > `pip install -r requirements.txt`

   b. Using `conda`
    - Create environment from `.yml`:
      > `conda env create -f environment.yml`
    - Activate environment:
      > `conda activate mlspeech`

4. Run setup to export `PYTHONPATH`:

    `source setup.sh`

## 3. Script execution

1. To create the datasets, run:

    > `python parsers/create_datasets.py`

    This will extract the features from both datasets, create the waveplots and spectrograms examples,
    and finally create the `train.csv` and `test.csv`.

    During the extraction process, we create extra records using the original audio files to also
    introduce noise and time stretch. We've also explored pitch shift with little success.

    To avoid "cheating", the `test.csv` will be comprised only of actresses whose audio clips won't be involved in the training process. They will essentially be completely unknown data. The `test.csv` will not be included at any part of the training.

2. To create some generic plots, execute:
   > `python plots/general.py`

3. We use the `train.csv` exported in the previous step to find the features that seem most important
   for our use case. We explored this using 3 different methods:
   - `PCA`:
     > `python feature_engineering/pca.py`
   - `RFE` (explores for `linear` kernel of `SVM`):
     > `python feature_engineering/rfe.py`
   - `RandomForestClassifier` for feature importance:
     > `python feature_engineering/feature_importance.py`

   We save the list of most important features in a `pickle` file to use during model training.

   The best results were achieved by using the feature importance from `RandomForestClassifier`.

4. Model training & evaluation

   - We start by tuning the `SVM` hyperparameters using `GridSearchCV`:
     > `python models/svm_tuning.py`

   - Using the hyperparameters (`C` and `gamma` for `rbf` kernel) proposed by `GridSearchCV`,
     we move on to the `SVM` model training:

      > `python models/svm.py`

   - Similariy, we execute the script for the `KNN` classifier:

      > `python models/knn.py`

   **Trained models are exported to a joblib file for use in the demo.**

5. To test our model for both `SVM` and `KNN`, we just need to run the demo:
    > `python demo/demo.py`
