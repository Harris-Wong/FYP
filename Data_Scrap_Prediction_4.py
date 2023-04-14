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

def Decision(mlr, cnnlong, cnnshort, adjclose, rnn):
  #mlr = 6/4 prediction (float)
  #adjclose = 5/4 adj close (float)
  #rnn = array (float)
  if mlr >= adjclose + 450:
    return True
  else:
    return False

def calculate_conf_level(mlr,adjclose,news):
    if news:
        rnn = max(news)
    else:
        rnn = None
    dif_true_up = joblib.load("./trained_parameters/dif_true_up.save")
    dif_true_down = joblib.load("./trained_parameters/dif_true_down.save")
    mlr_dif = (mlr/adjclose)-1.0244
    mlr_dif = mlr_dif/1.0244
    if rnn:
        rnn_dif = (rnn-0.602)/0.602
    else:
        rnn_dif = 0
    target = 0
    if (mlr_dif > 0) or (rnn_dif > 0): # Buy signal
        target = max(mlr_dif,rnn_dif)
        p = sum(i < target for i in dif_true_up)
        cdf = p/len(dif_true_up)
        # print(f"Confidence level of this Buy signal is: {round(cdf*100,3)}%")
        return round(cdf,5)
        
    else:
        target = min(mlr_dif,rnn_dif)
        p = sum(i > target for i in dif_true_down)
        cdf = p/len(dif_true_down)
        # print(f"Confidence level of this Sell signal is: {round(cdf*100,3)}%")
        return round(cdf,5)
    
    
if __name__ == "__main__":
    json_file = {}
    for coin in assets:
        lstm2014 = 0
        lstm2017 = 0
        # Read csv
        df = pd.read_csv(f"data/processed/{coin}_RNN_CNN_predicted.csv",index_col=0)
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
        # Strategy
        news_array = str(df.iloc[-1,0])[1:-1].split(',')
        # Confidence Level
        for i in range(len(news_array)):
            number = news_array[i].split('.')
            number = float(number[1])/10.0
            news_array[i] = number
        cdf = calculate_conf_level(mlr_prediction,df.iloc[-1,3],news_array)
        # Strategy
        signal = Decision(mlr_prediction,df.iloc[-1,1],df.iloc[-1,2],df.iloc[-1,3],news_array)
        json_file[coin] = {"signal":signal,"conf":cdf}
        # Save CSV
        df.to_csv(f"data/processed/{coin}_final_predicted.csv")
        #print(f"{coin}: Done")
    # Save JSON
    json_object = json.dumps(json_file)
    # Writing to sample.json
    with open("signals.json", "w") as outfile:
        outfile.write(json_object)
    # Done, exit()
    exit()