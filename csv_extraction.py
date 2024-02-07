import os
import pandas as pd
import numpy as np

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
mean_y = []
std_y = []
mean_mfcc = []
std_mfcc = []
mean_sc = []
std_sc = []
mean_root_mean_square = []
std_root_mean_square = []
zero_crossing_rate = []
std_zero_crossing_rate = []

k=0
feature_df = pd.DataFrame({'mfcc1',
                          'mfcc2',
                          'mfcc3',
                          'mfcc4',
                          'mfcc5',
                          'mfcc6',
                          'mfcc7',
                          'mfcc8',
                          'mfcc9',
                          'mfcc10',
                          'mfcc11',
                          'mfcc12',
                          'mfcc13',
                          'mfcc14',
                          'mfcc15',
                          'mfcc16',
                          'mfcc17',
                          'mfcc18',
                          'mfcc19',
                          'mfcc20'})

# Iterate through files
for file_path in data_ravdess_paths:
    # Extract information from the filename
    filename = os.path.basename(file_path)
    parts = filename.split('-')
    inst,sr = librosa.load(file_path)
    
    #Mean of waveform
    mesos_y = np.mean(inst)
    standard_deviation_y = np.std(inst)

    #mfcc
    mfcc = librosa.feature.mfcc(y=inst)
    mesos_mfcc = np.mean(mfcc)
    standard_deviation_mfcc = np.std(mfcc)
    mfcc_array = np.mean(librosa.feature.mfcc(y=inst).T,axis=0)

    #central centroid
    sc = librosa.feature.spectral_centroid(y=inst,sr=sr)
    mesos_sc = np.mean(sc)
    standard_deviation_sc = np.std(sc)

    #rms
    rms = librosa.feature.rms(y=inst)
    mesos_rms = np.mean(rms)
    standard_deviation_rms = np.std(rms)

    #zcr
    zcr = librosa.feature.zero_crossing_rate(y=inst)
    mesos_zcr = np.mean(zcr)
    standard_deviation_zcr = np.std(zcr)

    # Example: 03(modality)-02(vocal channel)-01(emotion)-01(emotional intensity)-01(statement)-01(repetition)-01(actor_male_female)
    modality.append(int(parts[0]))
    vocal_channel.append(int(parts[1]))

    #change numerical to str 
    if int(parts[2]) == 1 :
        emotion.append('neutral')
    elif int(parts[2]) == 2 :
        emotion.append('calm')
    elif int(parts[2]) == 3 :
        emotion.append('happy')
    elif int(parts[2]) == 4 :
        emotion.append('sad')
    elif int(parts[2]) == 5 :
        emotion.append('angry')
    elif int(parts[2]) == 6 :
        emotion.append('fearful')
    elif int(parts[2]) == 7 :
        emotion.append('disgust')
    elif int(parts[2]) == 8 :
        emotion.append('surprised')

    emotional_intensity.append(int(parts[3]))
    statement.append(int(parts[4]))
    repetition.append(int(parts[5]))
    actor.append(int(parts[6].split('.')[0]))  # Removing the '.wav' extension
    mean_y.append(mesos_y)
    std_y.append(standard_deviation_y)
    mean_mfcc.append(mesos_mfcc)
    std_mfcc.append(standard_deviation_mfcc)
    mean_sc.append(mesos_sc)
    std_sc.append(standard_deviation_sc)
    mean_root_mean_square.append(mesos_rms)
    std_root_mean_square.append(standard_deviation_rms)
    zero_crossing_rate.append(mesos_zcr)
    std_zero_crossing_rate.append(standard_deviation_zcr)

    for i in range(1, 21):
        col_name = 'mfcc{}'.format(i)
        feature_df.loc[k, col_name] = mfcc_array[i-1]

    k=k+1

    gender = int(parts[6].split('.')[0])
    #If the actor is odd number then is male and even number is female, respectively
    if gender%2 == 1 :
        genders.append('Male')
    else :
        genders.append('Female')

    # Append information to lists
    file_paths.append(file_path)
    sample_rates.append(sr)

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
    'Gender': genders,
    'Mean waveform':mean_y,
    'Std of waveform':std_y,
    'Mean mfcc':mean_mfcc,
    'Std of mfcc':std_mfcc,
    'Mean spectral centroid':mean_sc,
    'Std of Spectral centroid':std_sc,
    'Root mean square':mean_root_mean_square,
    'Std of Root mean square':std_root_mean_square,
    'Mean zero crossing rate':zero_crossing_rate,
    'Std of zero crossing rate':std_zero_crossing_rate,
})

# Save the dataframe to a CSV file
df.to_csv(r'C:\Users\Mihalis\Desktop\ML\numerical_data_ravdess.csv', index=False)
print(feature_df)