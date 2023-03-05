#!/usr/bin/env python
# coding: utf-8

# ## Web App Section for Sentiment Model

# This is the companion app that will serve as the UI for the sentiment model. Objectives of this are as follows:
#    - Have easy to access UI
#    - Practice hosting a web app
#    - Practice importing my own functions from external sources
# <br><br>Using this as template: https://docs.streamlit.io/library/get-started/create-an-app

# This app is more compatible with a .py file. Converting to that.

# to run change directory to folder location of app, then [steamlit run <appname>.py]

# hosted on streamlit here: tatyanasasynuik/sentiment_webapp/main/Sentiment_WebApp.py

import streamlit as st
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import re
from my_sentiment_loader import load_models, predict, preprocess, stopwordlist, emojis

# st.write(os.listdir(os.curdir))
# st.write(path)

#### LOAD PICKLED MODELS ####
vectorizer, LSVCmodel, LRmodel, BNBmodel = load_models()


#### START APP DEV ####
st.title('Sentiment Analyzer')
st.subheader('A Web App built by Tatyana Sasynuik for the KaggleX Mentorship Cohort 2023')
#User Input
user_in = st.text_input("Enter text to be analyzed here.") #,max_chars=250
# user_in = 'testing this is getting annoying but I feel close! feeling myself'
# st.write(type(user_in))

#Start the clock
import time
t = time.time()

def diff_process(textdata):
    processedText = []
    textdata = [textdata]
    # textdata = [user_in] #have to wrap in brackets not list() or it will slice to letters
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for text in textdata:
        # print(text)
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
    
#chunk input up into a list and convert to lowercase
def process_data(text):
    # if word in stopwords:
    # lowercase = lambda x: str(text).lower()
    # lowered = lowercase(text)
    # final = lowered.split(" ")
    final = diff_process(text)
    return final

def generate_wordcloud(text):
    fig = plt.figure(figsize = (20,20))
    wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                   collocations=False).generate(" ".join(text))
    plt.imshow(wc)
    return st.pyplot(fig)

try:
    # st.text('Loading data...')
    if len(user_in) >1:
        #Process the data
        if st.checkbox('Show data'):
            st.subheader('Raw data')
            st.write(f'Raw User Input: "{user_in}"')
            st.subheader('Processing Output (stripped of stop words)')
            text_list = process_data(user_in)
            st.write(text_list)
        
        # st.subheader('Custom Processing Output')
        # st.write(user_in)
        # text_list2 = diff_process(user_in)
        # st.write(text_list)
        
        # st.subheader('Imported Processing Output')
        # st.write(user_in)
        # test_sample = preprocess(user_in)
        # st.write(test_sample)
except:
    st.write('Waiting for user input. Write something and get your sentiment score!')
    
try:
        st.subheader('Results from Bernoulli Naive Bayes Model')
        bnb_df = predict(vectorizer, BNBmodel, [user_in])
        st.write(bnb_df)
        
        st.subheader('Results from Linear Regression Model')
        LR_df = predict(vectorizer, LRmodel, [user_in])
        st.write(LR_df)
        
        # st.subheader('Results from Linear Support Vector Model')
        # svc_df = predict(vectorizer, LSVCmodel, text_list)
        # st.write(svc_df)
except:
    st.write('Something went wrong with the model!')
        
try:
        st.subheader('Word Cloud of Your Input')
        generate_wordcloud(text_list)
except:        
    st.write('Sorry! Something went wrong building the wordcloud.')


#timestamp the execution
st.write('Text Processing Complete.')
st.write(f'Time Taken: {round(time.time()-t,4)} seconds')