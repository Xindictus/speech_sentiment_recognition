import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score


data_path = (r'C:\Users\jimja\Desktop\Msc_AI\Εργασίες\Τρέχουσες\ML\speech_sentiment_recognition\rms_std_values.csv')
train_data = pd.read_csv(data_path)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit LabelEncoder and transform the 'y' column
train_data['label'] = label_encoder.fit_transform(train_data['label'])


# Splitting
X = train_data.drop('label', axis=1)
y = train_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN
knn_classifier = KNeighborsClassifier(n_neighbors=11, p=7, metric='euclidean')
knn_classifier.fit(X_train, y_train,)
y_pred = knn_classifier.predict(X_test)



cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))




