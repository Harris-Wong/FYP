#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import glob
import os
import numpy as np


# In[2]:


df_features = pd.read_json("./data/all_data.txt")


# In[3]:


for i in range(len(df_features.iloc[:,0])):
    df_features.iloc[i,0] = str(df_features.iloc[i,0])[:10]
    if type(df_features.iloc[i,0]) != str:
        df_features.iloc[i,0] = df_features.iloc[i,0].strftime("%d-%m-%Y")
    if (df_features.iloc[i,0][5] == "1"):
        df_features.iloc[i,0] = df_features.iloc[i,0][-2:]+"/"+df_features.iloc[i,0][5:7]+"/"+df_features.iloc[i,0][0:4]
    else :
        df_features.iloc[i,0] = df_features.iloc[i,0][-2:]+"/"+df_features.iloc[i,0][6]+"/"+df_features.iloc[i,0][0:4]
    if (df_features.iloc[i,0][0] == "0"):
        df_features.iloc[i,0] = df_features.iloc[i,0][1:]


# In[4]:


df_features.dropna(axis = 1, how = "all", inplace=True)


# In[5]:


def to_clean(df, m, n):

    df = df.apply(pd.to_numeric, errors='coerce')

    # If value = 0.00, then it is probably some troublesome data since stocks are bankrupt when 0
    df.iloc[:,6:].replace(0, np.nan, inplace=True)

    # Define fill NAN data to handle weekend and holidays
    for i in range(6, len(df.columns)):
        if not (df.iloc[:,i].isnull().values.any()):
            continue
        else:
            for j in range(len(df.iloc[:,i])):
                if pd.isnull(df.iloc[j,i]):
                    k = [0,0]
                    if j > 0  and j < len(df.iloc[:,i]) - 1:
                        # If previous date is not null
                        if not pd.isnull(df.iloc[j-1,i]):
                            k[0] = j-1
                        # Consecutive Null Period is within 2 weeks
                        if not pd.isnull(df.iloc[j+1,i]):
                            k[1] = j+1
                        elif ((j < len(df.iloc[:,i]) - 2) and not pd.isnull(df.iloc[j+2,i])):
                            k[1] = j+2
                        elif ((j < len(df.iloc[:,i]) - 3) and not pd.isnull(df.iloc[j+3,i])):
                            k[1] = j+3
                        elif ((j < len(df.iloc[:,i]) - 4) and not pd.isnull(df.iloc[j+4,i])):
                            k[1] = j+4
                        elif ((j < len(df.iloc[:,i]) - 5) and not pd.isnull(df.iloc[j+5,i])):
                            k[1] = j+5
                        elif ((j < len(df.iloc[:,i]) - 6) and not pd.isnull(df.iloc[j+6,i])):
                            k[1] = j+6
                        elif ((j < len(df.iloc[:,i]) - 7) and not pd.isnull(df.iloc[j+7,i])):
                            k[1] = j+7
                        elif ((j < len(df.iloc[:,i]) - 8) and not pd.isnull(df.iloc[j+8,i])):
                            k[1] = j+8
                        elif ((j < len(df.iloc[:,i]) - 9) and not pd.isnull(df.iloc[j+9,i])):
                            k[1] = j+9
                        elif ((j < len(df.iloc[:,i]) - 10) and not pd.isnull(df.iloc[j+10,i])):
                            k[1] = j+10
                        elif ((j < len(df.iloc[:,i]) - 11) and not pd.isnull(df.iloc[j+11,i])):
                            k[1] = j+11
                        elif ((j < len(df.iloc[:,i]) - 12) and not pd.isnull(df.iloc[j+12,i])):
                            k[1] = j+12
                        elif ((j < len(df.iloc[:,i]) - 13) and not pd.isnull(df.iloc[j+13,i])):
                            k[1] = j+13
                    # Linear interpolation between the last and next available data
                    if (k[0] > 0 and k[1] > 0):
                        df.iloc[j, i] = ((k[1]-j)*df.iloc[k[0], i] + (j-k[0])*df.iloc[k[1], i]) / (k[1] - k[0])
        #if not (i%20):                
            #print(str(round((m/n + float(i) / (len(df.columns)*n)) * 100, 2)) + " % done")

    for i in range(6, len(df.columns)):
        if not (df.iloc[:,i].isnull().values.any()):
            continue
        else:
            for j in range(len(df.iloc[:,i])-13, len(df.iloc[:,i])):
                if pd.isnull(df.iloc[j,i]):
                           df.iloc[j,i] = df.iloc[j-1,i]
    return df


# In[6]:


cryptos_df = []
cryptos = ["BTC", "ADA", "BCH", "BNB", "DOGE", "ETH", "FTT", "LINK", "OKB", "SOL"]

for i in range(len(cryptos)):
    string = "./data/" + cryptos[i]+"_data.txt"
    crypto_df = pd.read_json(string)
    cryptos_df.append(crypto_df)


# In[7]:


print("This process may take some time....")
outdir = "./backtest/data/"
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
outdir = "./backtest/data/processed"
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
for i in range(len(cryptos)):
    df = df_features.copy()
    
    adj_Close = cryptos_df[i]["Adj Close"].to_list()
    df.insert(loc=1, column='Adj Close', value=adj_Close)
    
    for p in range(1, 2):
        if not (df.iloc[:,p].isnull().values.any()):
            continue
        else:
            for j in range(len(df.iloc[:,p])-4, len(df.iloc[:,p])):
                if pd.isnull(df.iloc[j,p]):
                           df.iloc[j,p] = df.iloc[j-1,p]
    
    cryptos_df[i]["Adj Close"] = df["Adj Close"]
    Tmr_adjClose = cryptos_df[i]["Adj Close"][1:].to_list()
    Tmr_adjClose.append(np.nan)
                        
    df.insert(loc=2, column='Tmr_adjClose', value=Tmr_adjClose)

    Change = df['Tmr_adjClose'] - df["Adj Close"]
    df.insert(loc=3, column='Change in AdjClose', value=Change)

    percent = df['Change in AdjClose'] / df['Adj Close']
    df.insert(loc=4, column='Percentage Change', value=percent)

    volume = cryptos_df[i]["Volume"].to_list()
    df.insert(loc=5, column='Volume', value=volume)
    
    for p in range(5, 6):
        if not (df.iloc[:,p].isnull().values.any()):
            continue
        else:
            for j in range(len(df.iloc[:,p])-4, len(df.iloc[:,p])):
                if pd.isnull(df.iloc[j,p]):
                           df.iloc[j,p] = df.iloc[j-1,p]
                        
    pn = []
    for k in range(len(df)):
        if (df.iloc[k,4] >0):
            pn.append(1)
        else:
            pn.append(0)
    df.insert(loc=5, column='Positive/Negative', value=pn)
    df.set_index('Date', inplace=True)
    
    df = to_clean(df, i, len(cryptos))
    
    date_column = df.index.to_list()
    required_date_from = date_column[-538]
    
    predict_df = df.iloc[-538:].copy()
    today = predict_df.iloc[-1]
    today_index = predict_df.index[-1]
                        
    predict_df.dropna(axis = 1, thresh=(len(predict_df.index) - 3), inplace=True)
    
    # Drop rows if any NAN exists on that date
    predict_df_for_index = predict_df.copy()
    predict_df.dropna(axis = 0, how = "any", inplace=True)
    predict_df = predict_df.append(pd.DataFrame([today],index=[today_index],columns=predict_df.columns))
    store_name = "./backtest/data/processed/" + cryptos[i] +"_prediction.csv"
    predict_df.to_csv(store_name)
    
print("All done!")


# In[3]:


# In[9]:


outdir = "./backtest/data/processed/images"
if not os.path.exists(outdir):
    os.mkdir(outdir)


# In[10]:


def draw_images(crypto_name, crypto, crypto_prices, crypto_volume, crypto_close, exp1, exp2, macd, signal_line, k, m):

    spaths = []
    lpaths = []
    
    import matplotlib.pyplot as plt
    import matplotlib
    # Turn off interactive mode to speed up
    plt.ioff()

    import warnings
    warnings.filterwarnings("ignore")
    for i in range(len(crypto)-537, len(crypto)+1):
        for n in [12, 26]:
            df = crypto_prices.iloc[i-n: i]
            max_volume = np.asscalar(crypto_volume[i-26:i].max(axis=0).values)
            df_volume = crypto_volume[i-n:i]
            df_macd = macd[i-n:i]
            df_signal = signal_line[i-n:i]
            df_ema1 = exp1[i-n:i]
            df_ema2 = exp2[i-n:i]


            width  = 0.9   # width of real body
            width2 = 0.05  # width of shadow

            px = 1/plt.rcParams['figure.dpi']  # pixel in inches
            plt.subplots(figsize=(32*px, 32*px))
            fig, ax = plt.subplots(2,1, gridspec_kw={'height_ratios': [3, 1]})
            # find the rows that are bullish
            dfup = df[df.Close >= df.Open]
            # find the rows that are bearish
            dfdown = df[df.Close < df.Open]
            # plot the bullish candle stick
            #fig.tight_layout()
            ax[0].bar(dfup.index, dfup.Close - dfup.Open, width, 
                   bottom = dfup.Open, edgecolor='g', color='green')
            ax[0].bar(dfup.index, dfup.High - dfup.Close, width2, 
                   bottom = dfup.Close, edgecolor='g', color='green')
            ax[0].bar(dfup.index, dfup.Low - dfup.Open, width2, 
                   bottom = dfup.Open, edgecolor='g', color='green')
            # plot the bearish candle stick
            ax[0].bar(dfdown.index, dfdown.Close - dfdown.Open, width, 
                   bottom = dfdown.Open, edgecolor='r', color='red')
            ax[0].bar(dfdown.index, dfdown.High - dfdown.Open, width2, 
                   bottom = dfdown.Open, edgecolor='r', color='red')
            ax[0].bar(dfdown.index, dfdown.Low - dfdown.Close, width2, 
                   bottom = dfdown.Close, edgecolor='r', color='red')
            ax[0].axis("off")


            # Plot volume
            # Bounded by 0 and highest volume, linear scale
            ax[1].bar(df_volume.index, df_volume.Volume, width, color='grey')
            ax[1].set_ylim([0, max_volume])


            # Plot ema 12-day and 26-day
            ax[0].plot(df_ema1.index, df_ema1.iloc[:,0], color = 'orange')
            ax[0].plot(df_ema2.index, df_ema2.iloc[:,0], color = 'blue')

            # Outputting the corrsponding images
            plt.axis('off')

            d = df.index[-1].strftime("%Y%m%d")
            if n == 12:
                path = "./backtest/data/processed/images/" + crypto_name + d + "short.png"
                spaths.append(path)
            elif n == 26:
                path = "./backtest/data/processed/images/" + crypto_name + d + "long.png"
                lpaths.append(path)
            
            matplotlib.use('Agg')
            plt.savefig(path, format="png")
            plt.close(fig)
            plt.clf()
        print(str(round(k/m *100, 2)) +"% done")
    return spaths, lpaths


# In[11]:


print("This process may take some time....")
for k in range(len(cryptos)):
    cryptos_df[k].dropna(axis=0, how="any", inplace=True)
    start_date_of_crypto = cryptos_df[k]["Date"].iloc[0]
    for i in range(len(cryptos_df[k].iloc[:,0])):
        cryptos_df[k].iloc[i,0] = str(cryptos_df[k].iloc[i,0])[:10]
        if (cryptos_df[k].iloc[i,0][5] == "1"):
            cryptos_df[k].iloc[i,0] = cryptos_df[k].iloc[i,0][-2:]+"/"+cryptos_df[k].iloc[i,0][5:7]+"/"+cryptos_df[k].iloc[i,0][0:4]
        else :
            cryptos_df[k].iloc[i,0] = cryptos_df[k].iloc[i,0][-2:]+"/"+cryptos_df[k].iloc[i,0][6]+"/"+cryptos_df[k].iloc[i,0][0:4]
        if (cryptos_df[k].iloc[i,0][0] == "0"):
            cryptos_df[k].iloc[i,0] = cryptos_df[k].iloc[i,0][1:]
    cryptos_df[k].set_index('Date', inplace=True)
    cryptos_df[k]['Close'] = cryptos_df[k]['Adj Close']

    p = len(cryptos_df[k])
    
    
    # 12 and 26 day trend most common for calculating MACD
    crypto = cryptos_df[k].set_index(pd.date_range(start_date_of_crypto, periods=p, freq="d"))
    crypto_prices = crypto[["Open","High","Low","Close"]]
    
    for p in range(len(crypto_prices.columns)):
        if not (crypto_prices.iloc[:,p].isnull().values.any()):
            continue
        else:
            for j in range(len(crypto_prices.iloc[:,p])-4, len(crypto_prices.iloc[:,p])):
                if pd.isnull(crypto_prices.iloc[j,p]):
                           crypto_prices.iloc[j,p] = crypto_prices.iloc[j-1,-1]
    
    crypto_volume = crypto[["Volume"]]
    if (crypto_volume.iloc[:,0].isnull().values.any()):
        for j in range(len(crypto_volume.iloc[:,0])-4, len(crypto_volume.iloc[:,0])):
            if pd.isnull(crypto_volume.iloc[j,0]):
                       crypto_volume.iloc[j,0] = crypto_volume.iloc[j-1,0]
    
    crypto_close = crypto[["Adj Close"]]
    exp1 = crypto_close.ewm(span=12, adjust=False).mean()
    exp2 = crypto_close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    # Using 9-day ema of MACD as signal line is the norm
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    spaths, lpaths = draw_images(cryptos[k], crypto, crypto_prices, crypto_volume, crypto_close, exp1, exp2, macd, signal_line, k, len(cryptos))
    
    read_name = "./backtest/data/processed/" + cryptos[k] +"_prediction.csv"
    store_name = "./backtest/data/processed/" + cryptos[k] +"_imagepath.csv"
    labels_df = pd.read_csv(read_name)
    images_labels = labels_df.iloc[-538:,:]
    images_labels.insert(1,'RT_Short_Term_Candlesticks_Pathname',np.nan)
    images_labels["RT_Short_Term_Candlesticks_Pathname"] = spaths
    images_labels.insert(2,'RT_Long_Term_Candlesticks_Pathname',np.nan)
    images_labels["RT_Long_Term_Candlesticks_Pathname"] = lpaths
    images_labels.to_csv(store_name, index=False)
print("All done!")


# In[ ]:


# In[5]:


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
    string = "./backtest/data/processed/" + cryptos[i]+"_imagepath.csv"
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
    REQ_PATH = "./backtest/data/processed/images/" + cryptos[i]
    if not (os.path.isdir(REQ_PATH)):
        os.mkdir(REQ_PATH)
        os.mkdir(REQ_PATH + '/test_short')
        os.mkdir(REQ_PATH + '/test_long')
        os.mkdir(REQ_PATH + '/test_short/2')
        os.mkdir(REQ_PATH + '/test_long/2')
    path = "./backtest/data/processed/images/"

    all_files = glob.glob(os.path.join(path +"/*.png"))

    total_count = len(all_files)
    test_count = 0

    for path in all_files:
        if cryptos[i] not in path:
            continue
        filename = path[33:]
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
    o_string = "./backtest/data/processed/" + cryptos[i]+"_CNN_predicted.csv"
    cryptos_df[i].to_csv(o_string)


# In[ ]:





# In[44]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import praw
reddit = praw.Reddit(client_id='k26fEW5kz3fD6lx32iRqDQ', client_secret='yVGUF6DKD44M8_UR9rYyjZ98qR_F-Q', user_agent='Scraper')
import pandas as pd
import datetime
import os

posts = []
subreddit = reddit.subreddit('finance')
for post in subreddit.top(time_filter="all",limit=5000):
    posts.append([post.title, post.created])
    
posts = pd.DataFrame(posts,columns=['title', 'created'])

posts['created'] = ( pd.to_datetime(posts['created'],unit='s').dt.tz_localize('utc').dt.tz_convert('America/New_York'))
posts = posts.sort_values('created')

outdir = "./backtest/data/news"
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
posts.to_csv('./backtest/data/news/news3.csv', encoding = 'utf-8-sig') 
posts = pd.DataFrame(posts, columns = ['tiitle', 'created'])


# In[2]:


df_input = pd.read_csv("./data/news/input_news.csv")
df_input = df_input.rename(columns={'date': 'Date', 'input': 'News Header'})
df_input.set_index("Date", inplace=True)
df = pd.read_csv("./backtest/data/news/news3.csv")
df = df.iloc[::-1]
for i in range(len(df)):
    df["created"][i] = df["created"][i][:10]
df = df.rename(columns={'created': 'Date', 'title': 'News Header'})
df.set_index("Date", inplace=True)


# In[3]:


df.sort_index(ascending=True, inplace=True)
df['Date'] = df.index
for i in range(len(df.iloc[:,0])):
    if (df['Date'].iloc[i][5] == "1"):
        df['Date'].iloc[i] = df['Date'].iloc[i][-2:]+"/"+df['Date'].iloc[i][5:7]+"/"+df['Date'].iloc[i][0:4]
    else :
        df['Date'].iloc[i] = df['Date'].iloc[i][-2:]+"/"+df['Date'].iloc[i][6]+"/"+df['Date'].iloc[i][0:4]
    if (df['Date'].iloc[i][0] == "0"):
        df['Date'].iloc[i] = df['Date'].iloc[i][1:]
df.set_index('Date', inplace=True)

if 'Unnamed: 0' in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
df = df.append(df_input)


# In[4]:


cryptos_df = []
cryptos = ["BTC", "ADA", "BCH", "BNB", "DOGE", "ETH", "FTT", "LINK", "OKB", "SOL"]

for i in range(len(cryptos)):
    string = "./backtest/data/processed/" + cryptos[i]+"_CNN_predicted.csv"
    crypto_df = pd.read_csv(string)
    crypto_df = crypto_df.rename(columns={'Unnamed: 0.1': 'Date'})
    crypto_df.set_index('Date', inplace=True)
    cryptos_df.append(crypto_df)


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nltk
from nltk import word_tokenize
from torchtext.data import Field, TabularDataset, BucketIterator


# In[6]:


# The RNN model
class Text_RNN(nn.Module):
    def __init__(self, n_vocab, embedding_dim, n_hidden, n_layers, dropout):
        super(Text_RNN, self).__init__()
        self.emb = nn.Embedding(n_vocab, embedding_dim) 
        self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                dropout=dropout, 
                batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sent, sent_len):
        # sent: batch_size, max_sent_len
        # sent_len: batch_size
        sent_emb = self.emb(sent)  #batch_size, max_sent_len, embedding_dim
        outputs, h_n = self.rnn(sent_emb)
        outputs = self.dropout(h_n)
        outputs = self.fc(outputs)
        out = self.sigmoid(outputs)
        return out


# In[7]:


# Define saving and loading of models
def save_checkpoint(save_path, model, optimizer, val_acc):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_acc': val_acc}

    torch.save(state_dict, save_path)
    print(f'Model saved to {save_path}')

def load_checkpoint(save_path, model, optimizer):
    save_path = save_path 
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    val_acc = state_dict['val_acc']
    print(f'Model loaded from {save_path}')
    return val_acc


# In[8]:


# LSTM model
class Text_RNN_m2(nn.Module):
    def __init__(self, n_vocab, embedding_dim, n_hidden, n_layers, dropout):
        super(Text_RNN_m2, self).__init__()
        self.emb = nn.Embedding(n_vocab, embedding_dim) 
        self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=n_hidden,
                num_layers=n_layers,
                dropout=dropout, 
                batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sent, sent_len):
        sent_emb = self.emb(sent) 
        outputs, (h_n, c_n) = self.lstm(sent_emb)
        outputs = self.dropout(h_n)
        outputs = self.fc(outputs)
        out = self.sigmoid(outputs)
        return out


# In[9]:


def TEST_non_ensemble(model, test_loader, device): 
    model.to(device)
    y_pred = []
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(test_loader):
            sent, sent_len = batch.news
            inputs_sent = sent.to(device)
            inputs_sent_len = sent_len.to(device)
            # Make Prediction
            outputs = model(inputs_sent, inputs_sent_len)
            outputs = outputs.reshape(-1,1)
            for predicted in outputs:
                y_pred.append(predicted.item())
    return y_pred

def TEST_ensemble(model_1_tuple, model_2_tuple, model_3_tuple, test_loader, device, average_cal): 
    model_1 = model_1_tuple[0].to(device)
    model_2 = model_2_tuple[0].to(device)
    model_3 = model_3_tuple[0].to(device)
    y_pred = []
    with torch.no_grad():
        model_1.eval()
        model_2.eval()
        model_3.eval()
        for i, batch in enumerate(test_loader):
            sent, sent_len = batch.news
            inputs_sent = sent.to(device)
            inputs_sent_len = sent_len.to(device)
            
            # Make Prediction
            outputs_1 = model_1(inputs_sent, inputs_sent_len)
            outputs_1 = outputs_1.reshape(-1,1)
            outputs_2 = model_2(inputs_sent, inputs_sent_len)
            outputs_2 = outputs_2.reshape(-1,1)
            outputs_3 = model_3(inputs_sent, inputs_sent_len)
            outputs_3 = outputs_3.reshape(-1,1)

            if average_cal == "Simple average":
                outputs = (outputs_1 + outputs_2 +outputs_3) / float(3.0)
            elif average_cal == "Weighted average":
                w1, w2, w3 = weighted_average(model_1_tuple, model_2_tuple, model_3_tuple)
                outputs = outputs_1*w1 + outputs_2*w2 + outputs_3*w3
            for predicted in outputs:
                y_pred.append(predicted.item())
    return y_pred


# In[46]:


# In[10]:


from torch.optim import Adam
for i in range(len(cryptos)):
    df_pred = cryptos_df[i][["Adj Close"]]
    #try_merge = pd.merge(df, df_pred, how="inner", on=["Date"])
    
    
    
    df_merge = pd.merge(df, df_pred, how="inner", on=["Date"])
    
    
    if 'Unnamed: 0' in  df_merge.columns:
        df_merge.drop(columns=["Unnamed: 0"], inplace=True)
    prediction_column = df_merge[['Adj Close', 'News Header']]
    
    outcome_array = []
    
    for n in range(len(cryptos_df[i])):
        prediction_date = cryptos_df[i].index.to_list()[n]
        
        if prediction_date not in df_merge.index:
            predicted = [np.nan]
            outcome_array.append(predicted)
        
        else:
            prediction_row = prediction_column.loc[[prediction_date]]
            test_df = prediction_row
            test_df.rename(columns={"Adj Close": "label", "News Header": "news"}, inplace=True)
            
            test_df['label'] = test_df['label'].astype('int')
            PATH = "./backtest/data/news/"
            test_df.to_csv(PATH + "news_prediction.csv", index=False)

            txt_field = Field(tokenize=word_tokenize, lower=True, batch_first=True, include_lengths=True) 
            label_field = Field(sequential=False, use_vocab=False, batch_first=True)

            train = TabularDataset(path='./data/news/training.csv', format='csv', fields=[('label', label_field), ('news', txt_field)], skip_header=True)
            test = TabularDataset(path=PATH + 'news_prediction.csv', format='csv', fields=[('label', label_field), ('news', txt_field)], skip_header=True)
            txt_field.build_vocab(train, min_freq=2)

            # Build into the vocabulary from distinct words
            vocab_distinct = set(txt_field.vocab.itos)

            # Make a new object instance for LSTM model
            model_2 = Text_RNN_m2(n_vocab=len(vocab_distinct), embedding_dim=50, n_hidden=64, n_layers=1, dropout=0.1).cuda()

            save_name_model_2 = 'LSTM model'
            path = "./trained_parameters/" + save_name_model_2
            criterion = nn.BCELoss()
            optimizer = Adam(model_2.parameters())
            load_checkpoint(path, model_2, optimizer)

            test_iter = BucketIterator(test, batch_size=1, shuffle=False)

            device = 'cpu'
            best_model = model_2
            predicted = TEST_non_ensemble(best_model, test_iter, device)
            another_array = []
            for pred in range(len(predicted)):
                another_array.append(predicted[pred])
            outcome_array.append(another_array)
    
    cryptos_df[i].insert(0, "RNN_Prediction", outcome_array)
    
    df_output = cryptos_df[i]

    if 'Unnamed: 0' in  df_output.columns:
        df_output.drop(columns=["Unnamed: 0"], inplace=True)
    if 'Unnamed: 0.1' in  df_output.columns:
        df_output.drop(columns=["Unnamed: 0.1"], inplace=True)
    o_string = "./backtest/data/processed/" + cryptos[i]+"_RNN_CNN_predicted.csv"
    df_output.to_csv(o_string)


# In[1]:


import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.linear_model import LinearRegression
import json
import warnings
import os

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
# load trained models and scalers
sc2014 = joblib.load("./trained_parameters/Scaler/sc2014_updated.save")
sc2017 = joblib.load("./trained_parameters/Scaler/sc2017_updated.save")
# model2014 = keras.models.load_model("models/lstm_model2014_updated/")
# model2017 = keras.models.load_model("models/lstm_model2017_updated/")
loaded_2014 = open('./trained_parameters/models/model_2014.json', 'r')
model2014 = keras.models.model_from_json(loaded_2014.read())
model2014.load_weights("./trained_parameters/models/model2014.h5")
loaded_2017 = open('./trained_parameters/models/model_2017.json', 'r')
model2017 = keras.models.model_from_json(loaded_2017.read())
model2017.load_weights("./trained_parameters/models/model2017.h5")
features2014 = ["Adj Close"] + list(sc2014.feature_names_in_)
features2017 = ["Adj Close"] + list(sc2017.feature_names_in_)
assets = ['ADA','BCH','BNB','BTC','DOGE','ETH','FTT','LINK','OKB','SOL']
# create batches for LSTM
def create_lstm_dataset(dataset,y_index,timestep=1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-timestep):
        a = dataset.iloc[i:(i+timestep),:]
        dataX.append(a)
        dataY.append(dataset.iloc[i+timestep,y_index])
    return np.array(dataX),np.array(dataY)

def classify_lstm(prediction):
    c = [0]*16
    for i in range(1,len(prediction)):
        if prediction[i] > prediction[i-1]:
            c.append(1)
        else:
            c.append(0)
    return c


# In[2]:


def Decision(mlr, cnnlong, cnnshort, adjclose, rnn):
    #mlr = 6/4 prediction (float)
    #adjclose = 5/4 adj close (float)
    #rnn = array (float)
    #maxrnn = 0
    #if rnn:
    #    maxrnn = max(rnn)
    #if (mlr/adjclose>1.0224) | (maxrnn > 0.602):
    #    return True
    #else:
    #    return False
    rnn = np.array(news).astype(np.float)
    if news:
        maxrnn = rnn.mean()
        if ((mlr/adjclose - 1.0224)/1.0224 + (maxrnn - 0.602)/0.602) >0:
            return True
        else:
            return False
    else:
        if (mlr/adjclose - 1.0224) >0:
            return True
        else:
            return False

def calculate_conf_level(mlr,adjclose,news,coin):
    if news:
        news = np.array(news).astype(np.float)
        rnn = news.mean()
        best_rnn = max(news)
    else:
        rnn = None
    dif_true_up = joblib.load(f"./trained_parameters/conf_intervals/{coin}_dif_true_up.save")
    dif_true_down = joblib.load(f"./trained_parameters/conf_intervals/{coin}_dif_true_down.save")
    mlr_dif = (mlr/adjclose)-1.0244
    mlr_dif = mlr_dif/1.0244
    
    if rnn:
        rnn_dif = (rnn-0.602)/0.602
        best_rnn_dif = (best_rnn-0.602)/0.602
        total_dif = (mlr_dif + rnn_dif) /2
    else:
        best_rnn_dif = 0
        total_dif = mlr_dif
        
    target = 0
    #if (mlr_dif > 0) or (best_rnn_dif > 0): # Buy signal
    if total_dif > 0:
        target = total_dif
        p = sum(i < target for i in dif_true_up)
        cdf = p/len(dif_true_up)
        # print(f"Confidence level of this Buy signal is: {round(cdf*100,3)}%")
        return round(cdf,5)
        #else:
            #cdf = 0
        
    else:
        #if total_dif < 0:
        target = total_dif
        p = sum(i > target for i in dif_true_down)
        cdf = p/len(dif_true_down)
        # print(f"Confidence level of this Sell signal is: {round(cdf*100,3)}%")
        return round(cdf,5)
        #else:
            #cdf = 0


# In[12]:


if __name__ == "__main__":
    json_file = {}
    for coin in assets:
        df_all = pd.read_csv(f"backtest/data/processed/{coin}_RNN_CNN_predicted.csv",index_col=0)
        df_all.insert(0, "MLR_Prediction", np.nan)
        signals =[]
        confs= []
        json_file[coin] = {}
        for n in range(26, len(df_all)):
            df = df_all.iloc[n-26:n]
            lstm2014 = 0
            lstm2017 = 0
            # Read csv
            df2014 = df.copy()[features2014]
            df2017 = df.copy()[features2017]
            # Normalize df
            df2014.iloc[:,1:] = sc2014.transform(df2014.iloc[:,1:])
            df2017.iloc[:,1:] = sc2017.transform(df2017.iloc[:,1:])
            # Create LSTM batches
            x2014,y2014 = create_lstm_dataset(df2014,0,15)
            x2017,y2017 = create_lstm_dataset(df2017,0,15)
            # LSTM Prediction
            p_2014 = model2014.predict(x2014,verbose=0)
            p_2017 = model2017.predict(x2017,verbose=0)
            p_2014 = p_2014.reshape(1,-1)[0]
            p_2017 = p_2017.reshape(1,-1)[0]
            c_2014 = classify_lstm(p_2014)
            c_2017 = classify_lstm(p_2017)
            df['LSTM_2014'] = c_2014
            df['LSTM_2017'] = c_2017
            # MLR fit
            x = df[['LSTM_2014','LSTM_2017','CNN_Prediction_long','CNN_Prediction_short']][-26:-1]
            y = df[['Adj Close']][-26:-1]
            mlr = LinearRegression().fit(x,y)
            mlr_prediction = mlr.predict(df[['LSTM_2014','LSTM_2017','CNN_Prediction_long','CNN_Prediction_short']].iloc[-1:,:])[0][0]
            #df.insert(0, "MLR_Prediction", np.nan)
            news_array = []
            # Strategy
            df_all.iloc[n, 0] = mlr_prediction
            #df.iloc[-1, 0] = mlr_prediction
            if df.iloc[-1,1] == "[nan]":
                news_array = []
            else:
                news_array = str(df.iloc[-1,1])[1:-1].split(',')
                for i in range(len(news_array)):
                    news_array[i] = float(news_array[i])
            # Confidence Level
            cdf = calculate_conf_level(mlr_prediction,df.iloc[-1,4],news_array, coin)
            # Strategy
            signal = Decision(mlr_prediction,df.iloc[-1,2],df.iloc[-1,3],df.iloc[-1,4],news_array)
            signals.append(signal)
            confs.append(cdf)
            json_file[coin][df.index.to_list()[-1]] = {"signal":signal,"conf":cdf}
            
        # Save CSV
        df_all = df_all.iloc[26:]
        df_all.insert(0, "Confidnece", confs)
        df_all.insert(0, "Signals", signals)
        df_all.to_csv(f"backtest/data/processed/{coin}_final_predicted.csv")
        #print(f"{coin}: Done")
    # Save JSON
    json_object = json.dumps(json_file)
    # Writing to sample.json
    with open("backtest_signals.json", "w") as outfile:
        outfile.write(json_object)
    # Done, exit()
    #exit()


# In[ ]:




