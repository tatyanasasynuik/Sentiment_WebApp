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
from my_sentiment_loader import load_models, predict, preprocess, path

# st.write(os.listdir(os.curdir))
# st.write(path)

#### LOAD PICKLED MODELS ####
vectorizer, LSVCmodel, LRmodel, BNBmodel = load_models()


#### START APP DEV ####
st.title('Sentiment Analyzer')
st.subheader('A Web App built by Tatyana Sasynuik for the KaggleX Mentorship Cohort 2023')
#User Input
user_in = st.text_input("Enter text to be analyzed here. (Limited to 250 characters for speed)", max_chars = 250)

try:
    if len(user_in) >1:
        st.write(f'Raw User Input: "{user_in}"')

        def process_data(text):
            lowercase = lambda x: str(text).lower()
            return lowercase
        
        st.text('Loading data...')
        data = process_data(user_in)
        # st.text("Done! (using st.cache)")
        
        import time
        t = time.time()
        processedtext = preprocess(user_in)
        st.write(f'Text Preprocessing complete.')
        st.write(f'Time Taken: {round(time.time()-t)} seconds')
        
        # if st.checkbox('Show raw data'):
        #     st.subheader('Raw data')
        #     st.write(data)
        
        st.subheader('Results from Bernoulli Naive Bayes Model')
        bnb_df = predict(vectorizer, BNBmodel, user_in)
        st.write(bnb_df)
        
        # st.subheader('Results from Linear Support Vector Model')
        # svc_df = predict(vectorizer, LSVCmodel, user_in)
        # st.write(svc_df)
        
        
        st.subheader('Results from Linear Regression Model')
        LR_df = predict(vectorizer, LRmodel, user_in)
        st.write(LR_df)
        
        
        st.subheader('Word Cloud of Your Input')
        
        def generate_wordcloud(text):
            fig = plt.figure(figsize = (20,20))
            wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
                           collocations=False).generate(" ".join(text))
            plt.imshow(wc)
            return st.pyplot(fig)
        
        generate_wordcloud(user_in)
        
except:
    st.write('Waiting for user input. Write something and get your sentiment score!')
    