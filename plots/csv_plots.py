import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv("mfccs_std_values.csv")
#df= pd.read_csv("flatness_std_values.csv")
#df= pd.read_csv("cens_std_values.csv")
#df= pd.read_csv("stft_std_values.csv")
#df= pd.read_csv("cqt_std_values.csv")
#df= pd.read_csv("mel_std_values.csv")
#df= pd.read_csv("rms_std_values.csv")
#df= pd.read_csv("spectral_centroid_std_values.csv")
#df= pd.read_csv("spectral_bandwidth_std_values.csv")
#df= pd.read_csv("spectral_contrast_std_values.csv")
#df= pd.read_csv("spectral_flatness_std_values.csv")
#df= pd.read_csv("spectral_rolloff_std_values.csv")
#df= pd.read_csv("zero_crossing_rate_std_values.csv")




calm = df[('calm')]
happy = df[('happy')]
sad = df[('sad')]
angry = df[('angry')]
fear = df[('fear')]
disgust = df[('disgust')]
surprise = df[('surprise')]




plt.plot(calm)
plt.plot(happy)
plt.plot(sad)
plt.plot(angry)
plt.plot(fear)
plt.plot(disgust)
plt.plot(surprise)
plt.legend(["calm","happy","sad","angry","fear","disgust","surprise"])
plt.show()
