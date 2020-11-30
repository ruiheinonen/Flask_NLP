#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 20:15:59 2020

@author: fi000980702
"""
from flask import render_template
from flask import Flask
from flask_wtf import FlaskForm
from wtforms import TextAreaField , SubmitField
from wtforms.validators import DataRequired
from flask import request

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import models 
from tensorflow.keras import backend as K

import pickle

"""
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

import nltk 
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

import os
import re
import string
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
from textblob import Word
"""

app = Flask(__name__, template_folder='templates')
@app.route('/')
def home():
	return render_template('homepage.html')

"""
df = pd.read_csv('imdb_sentiment_data_2k.csv')

def clear_data(text):
    '''
    Remove unnecessary noise "\n" and replace it with an empty space
    Remove single characters replace them with an empty space
    Remove multiple spaces
    Remove html tags with BeautifulSoup
    '''   
    text = text.replace('\n', '')    
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ',text)
    text = re.sub(r'\s+', ' ', text)
    text = BeautifulSoup(text, 'html.parser').text
    return text

def lowercase(text):
    '''
    This ensures same words are treated equally e.g. "Banana" and "banana"
    '''
    return text.apply(lambda sentence: sentence.lower())

def remove_numbers(df):
    '''
    Numbers typically do not hold sentimental value therefore remove numbers,
    such as year, dates, times etc. However, in this particular setting, one 
    needs to be careful because in review data there could be sentences like
    "I give this movies a 5/5 rating"
    '''
    text = ''.join(word for word in df if not word.isdigit())
    return text

def remove_punctuations(text):
    '''
    Special characters, such as !@#? can create unwanted noise as they do not contribute much or at all
    to the sentiments. Therefore remove punctuations and replace them with an empty space
    '''
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def lemmatize(df):
    '''
    Lemmatize text so that same words are treated equally e.g. "bananas" becomes "banana"
    '''
    return df.apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))

def remove_stopwords(df):
    '''
    These are most commonly occuring words, such as "the", "is",
    which typically do not provide additional* information value for the model.
    '''
    stop_words = set(stopwords.words('english'))
    return df.apply(lambda x: [item for item in x.split() if item not in stop_words])

df['text'] = df['text'].apply(clear_data)                     # 
df['text'] = lowercase(df['text'])                            #
df['text'] = df['text'].apply(remove_punctuations)            # 
df['text'] = df['text'].apply(remove_numbers)                 # Apply all functions described above
df['text'] = lemmatize(df['text'])                            #
df['text'] = remove_stopwords(df['text'])                     #

#df2 = df.copy()
max_length = 300                                               #Cut of the text after this number of words
training_samples = 35000                                       # Trains on 35000 samples
validation_samples = 15000                                     # Validates on 15000 samples 
top_words = 13000                                              # Consider only the top 13000 words in the dataset 

tokenizer = Tokenizer(num_words=top_words)             # Creates a tokenizer configured to only account for top words
tokenizer.fit_on_texts(df['text'])                     # Builds the word index
sequences = tokenizer.texts_to_sequences(df['text'])   # Converts strings into lists of integer indices

X = pad_sequences(sequences, maxlen=max_length)        # Converts the lists of integers into a 2D integer tensor
y = np.asarray(df['sentiment'])                        # Converts the target label into numpy 
encoder = LabelEncoder()                               # 
encoder.fit(y)                                         # Binarize the target label
y = encoder.transform(y)                               #

X_train = X[:training_samples]                                    # Define train data matrix 
y_train = y[:training_samples]                                    # Define train label vector 
X_val = X[training_samples : training_samples+validation_samples]
y_val = y[training_samples : training_samples+validation_samples] 
"""

with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

@app.route('/predict',methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        K.clear_session()
              
        seq = loaded_tokenizer.texts_to_sequences(data)
        padded = pad_sequences(seq, maxlen=len(data) 
        model = models.load_model('sentiment_model.h5')
        pred = model.predict(padded)
        
        K.clear_session()
        
        return render_template('resultpage.html', prediction = pred)

if __name__ == '__main__':
	app.run(debug=True)
