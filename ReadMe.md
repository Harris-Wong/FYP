## Instructions to run the train/predict server program

### 1. "cd" to the directory

## Install necessary packages

### 2. Install necessary packages:

#### 2.0 Go to https://nodejs.org to install Node.js 
#### 2.1 Go to python.org/downloads to install Python version 3.8.10. If your machine has already installed Python with other version, such as Python 3.11, you must uninstall it. This can be done by going to the "Applications" of your local computer, find Python and uninstall it. Verify the correct version (Python 3.8.10) by typing in terminal / command prompt python --version OR python3 --version

### In the terminal / command prompt, type the following one by one.
#### 2.2 >npm install
#### 2.3 >pip install pandas==1.2.2 / >pip3 install pandas==1.2.2
#### 2.4 >pip install numpy==1.19.5 / >pip3 install numpy==1.19.5
#### 2.5 >pip install datetime / >pip3 install datetime
#### 2.6 >pip install matplotlib==3.3.4 / >pip3 install matplotlib==3.3.4
#### 2.7 >pip install tensorflow==2.4.1 / >pip3 install tensorflow==2.4.1
#### 2.8 >pip install keras==2.4.3 / pip3 install keras==2.4.3
#### 2.9 >pip install pyyaml h5py / >pip3 install pyyaml h5py
#### 2.10 >pip install nltk / >pip3 install nltk
#### 2.11 >pip install praw / >pip3 install praw
#### 2.12 Install pyTorch according to the guidelines listed here: https://pytorch.org/ or https://github.com/pytorch/pytorch#from-source
#### 2.13 >pip install torchtext==0.8.1 / >pip3 install torchtext==0.8.1


## Download trained model parameters (Too big can only through Google Drive).

### 3. Go to Google Drive to download CNN_trained_long and CNN_trained_short, replace those current folders under "trained parameters" directory

## Begin the server

### 4. >node app.js
