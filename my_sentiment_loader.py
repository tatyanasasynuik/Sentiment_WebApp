# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 15:20:24 2023

@author: TSasynu1
GOAL: Practice calling functions from external files
"""

# utilities
import re
import pickle
import pandas as pd
import os
import sys
from nltk.stem import WordNetLemmatizer


path = os.path.abspath(os.getcwd())


# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

# test_in = 'testing this is getting annoying but I feel close! feeling myself'

def preprocess(textdata):
   processedText = []
   
   # Create Lemmatizer and Stemmer.
   wordLemm = WordNetLemmatizer()
   
   # Defining regex patterns.
   urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
   userPattern       = '@[^\s]+'
   alphaPattern      = "[^a-zA-Z0-9]"
   sequencePattern   = r"(.)\1\1+"
   seqReplacePattern = r"\1\1"
   
   for text in textdata:
       text = text.lower()
       
       # Replace all URls with 'URL'
       text = re.sub(urlPattern,' URL',text)
       # Replace all emojis.
       for emoji in emojis.keys():
           text = text.replace(emoji, "EMOJI" + emojis[emoji])        
       # Replace @USERNAME to 'USER'.
       text = re.sub(userPattern,' USER', text)        
       # Replace all non alphabets.
       text = re.sub(alphaPattern, " ", text)
       # Replace 3 or more consecutive letters by 2 letter.
       text = re.sub(sequencePattern, seqReplacePattern, text)

       textwords = ''
       for word in text.split():
           # Checking if the word is a stopword.
           if word not in stopwordlist:
               # print(word)
               if len(word)>1:
                   # Lemmatizing the word.
                   word = wordLemm.lemmatize(word)
                   textwords += (word+' ')
       processedText.append(textwords)
       split = processedText[0].split(" ")
       while("" in split):
           split.remove("")
       return split 

# sample2 = preprocess(test_in)

def load_models():
    # Load the vectorizer.
    file = open(path+'/vectorizer-ngram-(1,2).pickle', 'rb')
    vectorizer = pickle.load(file)
    file.close()
    # Load SVC Model.
    file = open(path+'/LinearSVC.pickle', 'rb')
    LSVCmodel = pickle.load(file)
    file.close()
    # Load the LR Model.
    file = open(path+'/Sentiment-LR.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    # Load the BNB Model.
    file = open(path+'/Sentiment-BNB.pickle', 'rb')
    BNBmodel = pickle.load(file)
    file.close()
    
    return vectorizer, LSVCmodel, LRmodel, BNBmodel

def predict(vectorizer, model, text):
    # Predict the sentiment
    textdata = vectorizer.transform(text) #vectorizer.transform(preprocess(text))
    sentiment = model.predict(text)
    # sentiment = BNBmodel.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.
    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,1], ["Negative","Positive"])
    return df
