import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Read data
df = pd.read_csv(r'C:\Users\Mihalis\Desktop\ML\numerical_data_ravdess.csv')

labels = list(df['Emotion'].unique())

# Grab from actor 1 all emotions with modality=1, vocal channel=1, emotional intensity=1, statement=1, repetition=1
files = dict()
for i in range(len(labels)):
    tmp = df[df['Emotion'] == labels[i]][:1].reset_index()
    path = r'C:\Users\mihal\OneDrive\Desktop\ML\archive\Actor_01\03-01-0{}-01-01-01-01.wav'.format(i+1)
    files[labels[i]] = path

# Create a box plot for Emotional intensity
plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Emotional intensity', data=df)
plt.title('Box Plot of Mean MFCC for Each Emotion')
plt.show()

# Create a box plot for all means and std of extracted features
    
plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Mean waveform', data=df)
plt.title('Box Plot of Mean Waveform for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Std of waveform', data=df)
plt.title('Box Plot of Std Waveform for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Mean mfcc', data=df)
plt.title('Box Plot of Mean MFCC for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Std of mfcc', data=df)
plt.title('Box Plot of Std of MFCC for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Mean spectral centroid', data=df)
plt.title('Box Plot of Mean Spectral Centroid for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Std of Spectral centroid', data=df)
plt.title('Box Plot of Std Spectral centroid for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Root mean square', data=df)
plt.title('Box Plot of Root mean square for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Std of Root mean square', data=df)
plt.title('Box Plot of Std Root mean square for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Mean zero crossing rate', data=df)
plt.title('Box Plot of Zero Crossing Rate for Each Emotion')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Emotion', y='Std of zero crossing rate', data=df)
plt.title('Box Plot of Std Zero Crossing Rate for Each Emotion')
plt.show()

# Create a scatter plot with different colors for each class
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Mean mfcc', y='Mean spectral centroid', hue='Emotion', data=df, palette='bright')
plt.title('Scatter Plot of Mean MFCC vs Mean spectral centroid')
plt.xlabel('Mean MFCC')
plt.ylabel('Mean spectral centroid')
# Add a legend
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper right')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Mean mfcc', y='Root mean square', hue='Emotion', data=df, palette='bright')
plt.title('Scatter Plot of Mean MFCC vs Root mean square')
plt.xlabel('Mean MFCC')
plt.ylabel('Root mean square')
plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper right')
plt.show()

# Select relevant features for box plots
features = ['Mean waveform', 'Mean mfcc', 'Mean spectral centroid', 'Root mean square', 'Mean zero crossing rate']
# Set up subplots
fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(12, 12))
# Create box plots for each feature
for i, feature in enumerate(features):
    ax = axes[i]
    sns.boxplot(x='Emotion', y=feature, data=df, ax=ax)
    ax.set_title(f'Box Plot of {feature}')
    ax.set_xlabel('Emotion')
    ax.set_ylabel(feature)
# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# Select relevant features for box plots
features = ['Std of waveform', 'Std of mfcc', 'Std of Spectral centroid', 'Std of Root mean square','Std of zero crossing rate']
# Set up subplots
fig, axes = plt.subplots(nrows=len(features), ncols=1, figsize=(12, 12))
# Create box plots for each feature
for i, feature in enumerate(features):
    ax = axes[i]
    sns.boxplot(x='Emotion', y=feature, data=df, ax=ax)
    ax.set_title(f'Box Plot - {feature}')
    ax.set_xlabel('Emotion')
    ax.set_ylabel(feature)
# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()