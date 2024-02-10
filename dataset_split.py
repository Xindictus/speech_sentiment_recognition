import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn

import glob

import librosa
import librosa.display
import IPython.display as ipd
import random

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


# -------------bazw ta filepaths se ksexwristous fakelous analoga me to sunaisthma-------------


# Iterate through files
for file_path in data_ravdess_paths:
    # Extract information from the filename
    filename = os.path.basename(file_path)
    parts = filename.split('-')
    #emotion.append(int(parts[2]))

    # Load each audio file
    y, sr = librosa.load(file_path, sr=None)
    
    

    if parts[2] == '01':
            neutral.append(file_path)
    elif parts[2] == '02':
            calm.append(file_path)
    elif parts[2] == '03':
            happy.append(file_path)
    elif parts[2] == '04':
            sad.append(file_path)
    elif parts[2] == '05':
            angry.append(file_path)
    elif parts[2] == '06':
           fear.append(file_path)
    elif parts[2] == '07':
            disgust.append(file_path)
    elif parts[2] == '08':
            surprise.append(file_path)

#-------------ftiaxnw listes me to kommati kathe sunaisthmatos poy tha ginei train-------------

neutral_mix = []
calm_mix = []
happy_mix = []
sad_mix = []
angry_mix = []
fear_mix = []
disgust_mix = []
surprise_mix = []
#-------------xwrizw ta sunaisthmata poy tha ginoun train se ksexwtistous fakelous-------------

i = 0
while i <= 0.7*int(len(calm)):

      calm_mix.append(calm[i])
      happy_mix.append(happy[i])
      sad_mix.append(sad[i])
      angry_mix.append(angry[i])
      fear_mix.append(fear[i])
      disgust_mix.append(disgust[i])
      surprise_mix.append(surprise[i])
      i=i+1



#-------------kano combine tis listes train -------------
   
dataset_mix = calm_mix + happy_mix + sad_mix + angry_mix + fear_mix + disgust_mix + surprise_mix


#-------------kano shuffle to dataset pou prokuptei -------------


random.shuffle(dataset_mix)



#-------------xwrizw to dataset se train kai validation-------------


train_set = []
validation_set = []

k = 0
while k <= 0.7*int(len(dataset_mix)):

      train_set.append(dataset_mix[k])
      k = k+1

m = 0

dataset_mix.reverse()

while m <= 0.3*int(len(dataset_mix)):

      validation_set.append(dataset_mix[m])
      m = m+1





#-------------ftiaxnw listes me to kommati kathe sunaisthmatos poy tha ginei test-------------




neutral_test = []
calm_test = []
happy_test = []
sad_test= []
angry_test = []
fear_test = []
disgust_test = []
surprise_test = []

#--------------kanw reverse tis listes me ta sunaisthimata gia test--------------
neutral.reverse()
calm.reverse()
happy.reverse()
sad.reverse()
angry.reverse()
fear.reverse()
disgust.reverse() 
surprise.reverse() 

#-------------xwrizv ta sunaisthmata poy tha ginoun test se ksexwtistous fakelous-------------

j = 0

while j <=0.3*int(len(calm)):
    calm_test.append(calm[j])
    happy_test.append(happy[j])
    sad_test.append(sad[j])
    angry_test.append(angry[j])
    fear_test.append(fear[j])
    disgust_test.append(disgust[j])
    surprise_test.append(surprise[j])
    j=j+1


#-------------kano shuffle to test set pou prokuptei -------------


test_set = calm_test + happy_test + sad_test + angry_test + fear_test + disgust_test + surprise_test


print(len(train_set))
print(len(validation_set))
print(len(test_set))