#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shutil
from datetime import datetime as dt

# Reference: https://studymachinelearning.com/keras-imagedatagenerator-with-flow_from_directory/
import pandas as pd
import glob
import os

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Flatten, Dropout, Dense, RepeatVector
from tensorflow.keras.layers import LSTM


# In[2]:


cryptos_df = []
cryptos = ["BTC", "ADA", "BCH", "BNB", "DOGE", "ETH", "FTT", "LINK", "OKB", "SOL"]

for i in range(len(cryptos)):
    string = "./data/processed/" + cryptos[i]+"_imagepath.csv"
    crypto_df = pd.read_csv(string)
    
    df = crypto_df
    df["Date"] = crypto_df["Unnamed: 0"]
    df[['Day', 'Month', 'Year']] = df["Date"].str.split('/', n=2, expand=True)
    df['Day'] = [d if len(d) == 2 else '0'+d for d in df['Day']]
    df['Month'] = [m if len(m) == 2 else '0'+m for m in df['Month']]

    df['DateStr'] = df['Year'] + df['Month'] + df['Day']
    cryptos_df.append(df)


# In[5]:


all_test_sets = []
for i in range(len(cryptos)):
    REQ_PATH = "./data/processed/images/" + cryptos[i]
    if not (os.path.isdir(REQ_PATH)):
        os.mkdir(REQ_PATH)
        os.mkdir(REQ_PATH + '/test_short')
        os.mkdir(REQ_PATH + '/test_long')
        os.mkdir(REQ_PATH + '/test_short/2')
        os.mkdir(REQ_PATH + '/test_long/2')
    path = "./data/processed/images/"

    all_files = glob.glob(os.path.join(path +"/*.png"))

    total_count = len(all_files)
    test_count = 0

    for path in all_files:
        if cryptos[i] not in path:
            continue
        filename = path[24:]
        new_path = REQ_PATH

        test_count += 1
        total_count += 1
        if filename[-9] == 's':
            new_path = os.path.join(new_path, 'test_short')
        else:
            new_path = os.path.join(new_path, 'test_long')
        new_path = os.path.join(new_path, '2')
        new_path = os.path.join(new_path, filename)
        shutil.copyfile(path, new_path)

    test_sets = []
    for period in ('long', 'short'):
        print(REQ_PATH + '/test_'+period)
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_set = test_datagen.flow_from_directory(
            REQ_PATH + '/test_'+period,
            target_size=(256, 256),
            color_mode='rgb',
            batch_size=1,
            class_mode='binary',
            shuffle=False)
        test_sets.append(test_set)
    all_test_sets.append(test_sets)


# In[6]:


all_test_sets


# In[7]:


thresholds = pd.read_csv('./trained_parameters/threshold_CNN.csv')
import keras
from matplotlib import pyplot as plt
results = []
paths = ['./trained_parameters/CNN_trained_long', './trained_parameters/CNN_trained_short']
for i in range(len(cryptos)):
    test_sets = all_test_sets[i]
    for j in range(2):
        model_v = keras.models.load_model(paths[j])
        threshold = thresholds["Threshold"].iloc[j]
    
        filenames = test_sets[j].filenames
        nb_samples = len(filenames)
        predict = model_v.predict(test_sets[j], steps=nb_samples)
        pred_result = (predict >= threshold).astype(int).reshape(-1,1)
        result = np.concatenate((np.array(filenames).reshape(-1,1), pred_result), axis=1)
        date = [r[2:2+8] for r in result[:,0]]
        date = np.array(date)
        date = date.reshape(date.shape[0], 1)
        result = np.concatenate((result, date), axis=1)
        result = result[:, 1:]
        result = result[:, [1,0]]
        result = result[result[:,0].argsort()]
    
        df = pd.DataFrame({"Date" : result[:,0], "Prediction" : result[:,1]})
        results.append(df)


# In[9]:


for i in range(len(cryptos_df)):
    cryptos_df[i].insert(1,'CNN_Prediction_long',np.nan)
    cryptos_df[i].insert(2,'CNN_Prediction_short',np.nan)
    
    cryptos_df[i]['CNN_Prediction_long'] = results[i*2]['Prediction']
    cryptos_df[i]['CNN_Prediction_short'] = results[i*2+1]['Prediction']
    
    cryptos_df[i].drop(columns=['RT_Short_Term_Candlesticks_Pathname', 'RT_Long_Term_Candlesticks_Pathname', 'Date', 'Day', 'Month', 'Year', 'DateStr'], inplace=True)


# In[10]:


for i in range(len(cryptos_df)): 
    o_string = "./data/processed/" + cryptos[i]+"_CNN_predicted.csv"
    cryptos_df[i].to_csv(o_string)


# In[ ]:




