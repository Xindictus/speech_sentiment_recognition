import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,cohen_kappa_score,f1_score,roc_curve,roc_auc_score

data_path = (r'C:\Users\mihal\speech_sentiment_recognition\parsers\datasets\train.csv')
train_data = pd.read_csv(data_path)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit LabelEncoder and transform the 'y' column
train_data['label'] = label_encoder.fit_transform(train_data['label'])

# # Create a mapping between class labels and numerical representations
# class_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
# print("Mapping between class labels and numerical representations:")
# for class_label, numerical_value in class_mapping.items():
#     print(f"{class_label} -> {numerical_value}")


# Splitting
X = train_data.drop('label', axis=1)
y = train_data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# SVM
svm_classifier = SVC(kernel='linear',probability=True,random_state=1)
svm_classifier.fit(X_train, y_train,)
y_pred = svm_classifier.predict(X_test)

# METRICS
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred)

# Fit LabelBinarizer on y_true to transform it into a binary format
lb = LabelBinarizer()
lb.fit(y_test)

# Transform true labels and predicted probabilities
y_true_binarized = lb.transform(y_test)
y_probs_binarized = lb.transform(y_pred)

# Calculate ROC-AUC score for each class separately
roc_auc = roc_auc_score(y_true_binarized, y_probs_binarized, average='weighted')
# Plot ROC curves for each class
fpr = dict()
tpr = dict()
for i in range(len(lb.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs_binarized[:, i])
    plt.plot(fpr[i], tpr[i], label=f"ROC curve (class {lb.classes_[i]})")
# Calculate macro-average ROC curve
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(lb.classes_))]))

mean_tpr = np.zeros_like(all_fpr)
for i in range(len(lb.classes_)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

mean_tpr /= len(lb.classes_)
fpr_macro = all_fpr
tpr_macro = mean_tpr


# Plot macro-average ROC curve
plt.plot(fpr_macro, tpr_macro, label='macro-average ROC curve', linestyle=':', linewidth=4)
plt.show()
# Calculate micro and macro-average ROC-AUC score
roc_auc_macro = roc_auc_score(y_true_binarized, y_probs_binarized, average='macro')
roc_auc_micro = roc_auc_score(y_true_binarized, y_probs_binarized, average='micro')

# Print the calculated metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Cohen's Kappa:", kappa)
print("ROC-AUC Score (micro):", roc_auc_micro)
print("ROC-AUC Score (macro):", roc_auc_macro)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
classes = np.unique(y)
plt.xticks(classes, classes)
plt.yticks(classes, classes)
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()