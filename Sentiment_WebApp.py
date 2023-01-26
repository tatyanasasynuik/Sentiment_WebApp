#!/usr/bin/env python
# coding: utf-8

# ## Web App Section for Sentiment Model

# This is the companion app that will serve as the UI for the sentiment model. Objectives of this are as follows:
#    - Have easy to access UI
#    - Practice hosting a web app
#    - Practice importing my own functions from external sources
# <br><br>Using this as template: https://docs.streamlit.io/library/get-started/create-an-app

# This app is more compatible with a .py file. Converting to that.

import streamlit as st
import pandas as pd
import numpy as np

st.title('Test App: Hello!')


DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(10000)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Done! (using st.cache)")

st.subheader('Raw data')
st.write(data)