import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn

import glob

import librosa
import librosa.display
import IPython.display as ipd

data_path = (r'C:\Users\jimja\Desktop\Msc_AI\Εργασίες\Τρέχουσες\ML\speech_sentiment_recognition\archive\*\*.wav')
data_ravdess_paths = glob.glob(data_path)

# Initialize lists to store information
file_paths = []
neutral = []
calm = []
happy = []
sad = []
angry = []
fear = []
disgust = []
surprise = []





# Iterate through files
for file_path in data_ravdess_paths:
    # Extract information from the filename
    filename = os.path.basename(file_path)
    parts = filename.split('-')
    #emotion.append(int(parts[2]))

   # Load each audio file
    y, sr = librosa.load(file_path, sr=None)
    rms_raw= librosa.feature.rms(y=y)
    rms_cl = rms_raw[rms_raw != 0]
    rms_std = np.std(rms_cl)
    
    
    

    if parts[2] == '01':
            neutral.append(rms_std)
    elif parts[2] == '02':
            calm.append(rms_std)
    elif parts[2] == '03':
            happy.append(rms_std)
    elif parts[2] == '04':
            sad.append(rms_std)
    elif parts[2] == '05':
            angry.append(rms_std)
    elif parts[2] == '06':
           fear.append(rms_std)
    elif parts[2] == '07':
            disgust.append(rms_std)
    elif parts[2] == '08':
            surprise.append(rms_std)

'''
#plt.plot(neutral)
plt.plot(calm)
plt.plot(happy)
plt.plot(sad)
plt.plot(angry)
plt.plot(fear)
plt.plot(disgust)
plt.plot(surprise)
plt.legend(["calm","happy","sad","angry","fear","disgust","surprise"]) 
plt.show()

'''




df = pd.DataFrame({


'calm' :calm,
'happy' :happy,
'sad' :sad,
'angry' :angry,
'fear' :fear,
'disgust' :disgust,
'surprise' :surprise


})

#print(df)

df.to_csv(r"C:\Users\jimja\Desktop\Msc_AI\Εργασίες\Τρέχουσες\ML\speech_sentiment_recognition\rms_std_values.csv")