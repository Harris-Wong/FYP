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
for post in subreddit.top(time_filter="month",limit=1000):
    posts.append([post.title, post.created])
    
posts = pd.DataFrame(posts,columns=['title', 'created'])

posts['created'] = ( pd.to_datetime(posts['created'],unit='s').dt.tz_localize('utc').dt.tz_convert('America/New_York'))
posts = posts.sort_values('created')

outdir = "./data/news"
if not os.path.exists(outdir):
    os.mkdir(outdir)
    
posts.to_csv('./data/news/news3.csv', encoding = 'utf-8-sig') 
posts = pd.DataFrame(posts, columns = ['tiitle', 'created'])


# In[2]:


df_input = pd.read_csv("./data/news/input_news.csv")
df_input = df_input.rename(columns={'date': 'Date', 'input': 'News Header'})

for i in range(len(df_input)):
    df_input['Date'].iloc[i] = df_input['Date'].iloc[i][1:]
    if (df_input['Date'].iloc[i][0] == "0"):
        df_input['Date'].iloc[i] = df_input['Date'].iloc[i][1:]
    ind = (df_input)['Date'].iloc[i].index('/')
    if (df_input['Date'].iloc[i][ind+1] == "0"):
        df_input['Date'].iloc[i] = df_input['Date'].iloc[i][:ind+1] + df_input['Date'].iloc[i][ind+2:]


# In[3]:


df_input.set_index("Date", inplace=True)
df = pd.read_csv("./data/news/news3.csv")
df = df.iloc[::-1]
for i in range(len(df)):
    df["created"][i] = df["created"][i][:10]
df = df.rename(columns={'created': 'Date', 'title': 'News Header'})
df.set_index("Date", inplace=True)


# In[4]:


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


# In[5]:


cryptos_df = []
cryptos = ["BTC", "ADA", "BCH", "BNB", "DOGE", "ETH", "FTT", "LINK", "OKB", "SOL"]

for i in range(len(cryptos)):
    string = "./data/processed/" + cryptos[i]+"_CNN_predicted.csv"
    crypto_df = pd.read_csv(string)
    crypto_df = crypto_df.rename(columns={'Unnamed: 0.1': 'Date'})
    crypto_df.set_index('Date', inplace=True)
    cryptos_df.append(crypto_df)


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


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


# In[10]:


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


# In[11]:


from torch.optim import Adam
for i in range(len(cryptos)):
    df_pred = cryptos_df[i][["Adj Close"]]
    try_merge = pd.merge(df, df_pred, how="inner", on=["Date"])
    prediction_date = df_pred.index.to_list()[-1]
    df_merge = df_pred
    if prediction_date not in try_merge.index:
        df_merge = pd.merge(df, df_pred, how="outer", on=["Date"])

    else:
        df_merge = pd.merge(df, df_pred, how="inner", on=["Date"])
        
        
    if 'Unnamed: 0' in  df_merge.columns:
        df_merge.drop(columns=["Unnamed: 0"], inplace=True)
    prediction_column = df_merge[['Adj Close', 'News Header']]

    prediction_row = prediction_column.loc[[prediction_date]]
    
    test_df = prediction_row
    test_df.rename(columns={"Adj Close": "label", "News Header": "news"}, inplace=True)

    if prediction_date not in try_merge.index:
        predicted = [np.nan]
        test_df = test_df.iloc[[-1]]
        test_df['RNN_Prediction'] = predicted
        test_df = test_df[['RNN_Prediction']]
        
    else:
        test_df['label'] = test_df['label'].astype('int')
        PATH = "./data/news/"
        test_df.to_csv(PATH + "news_prediction.csv", index=False)

        txt_field = Field(tokenize=word_tokenize, lower=True, batch_first=True, include_lengths=True) 
        label_field = Field(sequential=False, use_vocab=False, batch_first=True)

        train = TabularDataset(path=PATH + 'training.csv', format='csv', fields=[('label', label_field), ('news', txt_field)], skip_header=True)
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
        test_df = test_df.iloc[[-1]]
        test_df['RNN_Prediction'] = [another_array]
        test_df = test_df[['RNN_Prediction']]
        


    df_output = pd.merge(test_df, cryptos_df[i], how="outer", on=["Date"])
    if 'Unnamed: 0' in  df_output.columns:
        df_output.drop(columns=["Unnamed: 0"], inplace=True)
    if 'Unnamed: 0.1' in  df_output.columns:
        df_output.drop(columns=["Unnamed: 0.1"], inplace=True)
    df_output = df_output.iloc[1:,].append(df_output.iloc[0,:])
    o_string = "./data/processed/" + cryptos[i]+"_RNN_CNN_predicted.csv"
    df_output.to_csv(o_string)


# In[ ]:




