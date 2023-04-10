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
        if not (i%10):
            print(str(round((m/n + float(i) / (len(df.columns)*n)) * 100, 2)) + " % done")

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
outdir = "./data/processed"
if not os.path.exists(outdir):
    os.mkdir(outdir)

for i in range(len(cryptos)):
    df = df_features.copy()
    adj_Close = cryptos_df[i]["Adj Close"].to_list()
    df.insert(loc=1, column='Adj Close', value=adj_Close)

    Tmr_adjClose = cryptos_df[i]["Adj Close"][1:].to_list()
    Tmr_adjClose.append(np.nan)
    df.insert(loc=2, column='Tmr_adjClose', value=Tmr_adjClose)

    Change = df['Tmr_adjClose'] - df["Adj Close"]
    df.insert(loc=3, column='Change in AdjClose', value=Change)

    percent = df['Change in AdjClose'] / df['Adj Close']
    df.insert(loc=4, column='Percentage Change', value=percent)

    volume = cryptos_df[i]["Volume"].to_list()
    df.insert(loc=5, column='Volume', value=volume)


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
    required_date_from = date_column[-15]

    predict_df = df.iloc[-15:].copy()
    today = predict_df.iloc[-1]
    today_index = predict_df.index[-1]
    predict_df.dropna(axis = 1, thresh=(len(predict_df.index) - 3), inplace=True)

    # Drop rows if any NAN exists on that date
    predict_df_for_index = predict_df.copy()
    predict_df.dropna(axis = 0, how = "any", inplace=True)
    predict_df = predict_df.append(pd.DataFrame([today],index=[today_index],columns=predict_df.columns))
    store_name = "./data/processed/" + cryptos[i] +"_prediction.csv"
    predict_df.to_csv(store_name)

print("All done!")


# In[8]:


outdir = "./data/processed/images"
if not os.path.exists(outdir):
    os.mkdir(outdir)


# In[9]:


def draw_images(crypto_name, crypto, crypto_prices, crypto_volume, crypto_close, exp1, exp2, macd, signal_line, k, m):

    spaths = []
    lpaths = []

    import matplotlib.pyplot as plt
    # Turn off interactive mode to speed up
    plt.ioff()

    import warnings
    warnings.filterwarnings("ignore")
    for i in range(len(crypto), len(crypto)+1):
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
                path = "./data/processed/images/" + crypto_name + d + "short.png"
                spaths.append(path)
            elif n == 26:
                path = "./data/processed/images/" + crypto_name + d + "long.png"
                lpaths.append(path)
            plt.savefig(path)
            plt.close(fig)
        print(str(round(k/m *100, 2)) +"% done")
    return spaths, lpaths


# In[10]:


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
    crypto_volume = crypto[["Volume"]]
    crypto_close = crypto[["Adj Close"]]
    exp1 = crypto_close.ewm(span=12, adjust=False).mean()
    exp2 = crypto_close.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    # Using 9-day ema of MACD as signal line is the norm
    signal_line = macd.ewm(span=9, adjust=False).mean()

    spaths, lpaths = draw_images(cryptos[k], crypto, crypto_prices, crypto_volume, crypto_close, exp1, exp2, macd, signal_line, k, len(cryptos))

    store_name = "./data/processed/" + cryptos[k] +"_imagepath.csv"
    labels_df = pd.read_csv(store_name)
    images_labels = labels_df.iloc[-16:,:]
    images_labels.insert(1,'RT_Short_Term_Candlesticks_Pathname',np.nan)
    images_labels["RT_Short_Term_Candlesticks_Pathname"].iloc[-1] = spaths[0]
    images_labels.insert(2,'RT_Long_Term_Candlesticks_Pathname',np.nan)
    images_labels["RT_Long_Term_Candlesticks_Pathname"].iloc[-1] = lpaths[0]
    images_labels.to_csv(store_name, index=False)
print("All done!")


# In[ ]:
