import numpy as np 
import pandas as pd
import soundfile as sf
import librosa
import librosa.display
from IPython.display import Audio
from tqdm import tqdm 
import os
import math
import json
import random
import librosa
import librosa.display
import IPython.display as ipd
from scipy.io import wavfile as wav
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import pickle


filename = 'randomforest_model.sav'
df_train = pd.read_csv('features_30_sec.csv')
X_train = df_train.drop(['label','filename','perceptr_mean','perceptr_var'],axis=1)
cols = X_train.columns
model = pickle.load(open(filename, 'rb'))
def main(filepath):
    audio_path = filepath
    x , sr = librosa.load(audio_path)
    print(type(x), type(sr))
    data_final=[]

    #length
    length = len(x)
    data_final.append(length)

    #chroma_stft_mean
    chroma_stft = librosa.feature.chroma_stft(x, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    #print(chroma_stft_mean)
    data_final.append(round(chroma_stft_mean,1))
    data_final.append(round(chroma_stft_var,1))

    #rms_mean
    rms = librosa.feature.rms(x)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    data_final.append(round(rms_mean,1))
    data_final.append(round(rms_var,1))

    #spectral_centroid_mean
    spectral_centroid = librosa.feature.spectral_centroid(x)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    data_final.append(round(spectral_centroid_mean,1))
    data_final.append(round(spectral_centroid_var,1))

    #spectral_bandwidth_mean
    spectral_bandwidth = librosa.feature.spectral_bandwidth(x)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var= np.var(spectral_bandwidth)
    data_final.append(round(spectral_bandwidth_mean,1))
    data_final.append(round(spectral_bandwidth_var,1))

    #rolloff_mean
    rolloff = librosa.feature.spectral_rolloff(x)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    data_final.append(round(rolloff_mean,1))
    data_final.append(round(rolloff_var,1))

    #zero_crossing_rate_mean
    zero_crossing_rate = librosa.feature.zero_crossing_rate(x)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)
    data_final.append(round(zero_crossing_rate_mean,1))
    data_final.append(round(zero_crossing_rate_var,1))

    #harmony
    harmony = librosa.effects.harmonic(x)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    #print(harmony_mean,harmony_var)
    data_final.append(round(harmony_mean,1))
    data_final.append(round(harmony_var,1))

    #tempo
    onset_env = librosa.onset.onset_strength(x, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
    #print(tempo)
    tempo=np.mean(tempo)
    #print(tempo)
    data_final.append(round(tempo,1))

    #MFCC
    mfccs = librosa.feature.mfcc(x, sr=sr)
    for mfcc in mfccs:
        mfcc_test = mfcc
        mfcc_mean=np.mean(mfcc_test) 
        mfcc_var=np.var(mfcc_test) 
        data_final.append(round(mfcc_mean,1))
        data_final.append(round(mfcc_var,1))

    #print(data_final)

    df_test=pd.DataFrame([data_final],columns = cols)
    prediction = model.predict(df_test)
    return(str(prediction[0]))
