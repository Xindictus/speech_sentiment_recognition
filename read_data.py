import os
import pandas as pd
import numpy as numpy

import matplotlib.pyplot as plt
import seaborn

import glob

import librosa
import librosa.display
import IPython.display as ipd

data_path = (r'C:\Users\Mihalis\Desktop\ML\archive\*\*.wav')
data_ravdess_paths = glob.glob(data_path)

# Initialize lists to store information
file_paths = []
sample_rates = []
modality = []
vocal_channel = []
emotion = []
emotional_intensity = []
statement = []
repetition = []
actor = []
genders = []

# Iterate through files
for file_path in data_ravdess_paths:
    # Extract information from the filename
    filename = os.path.basename(file_path)
    parts = filename.split('-')

    # Example: 03(modality)-02(vocal channel)-01(emotion)-01(emotional intensity)-01(statement)-01(repetition)-01(actor_male_female)
    modality.append(int(parts[0]))
    vocal_channel.append(int(parts[1]))
    emotion.append(int(parts[2]))
    emotional_intensity.append(int(parts[3]))
    statement.append(int(parts[4]))
    repetition.append(int(parts[5]))
    actor.append(int(parts[6].split('.')[0]))  # Removing the '.wav' extension
    gender = int(parts[6].split('.')[0])

    #If the actor is odd number then is male and even number is female, respectively
    if gender%2 == 1 :
        genders.append('Male')
    else :
        genders.append('Female')

    # Load each audio file
    y, sample_rate = librosa.load(file_path)

    # Append information to lists
    file_paths.append(file_path)
    sample_rates.append(sample_rate)

# Create a DataFrame
df = pd.DataFrame({
    'File Path': file_paths,
    'Sample Rate': sample_rates,
    'Modality': modality,
    'Vocal Channel': vocal_channel,
    'Emotion': emotion,
    'Emotional intensity': emotional_intensity,
    'Statement': statement,
    'Repetition': repetition,
    'Actor': actor,
    'Gender': genders
})

# Display the DataFrame
print(df.head(50))