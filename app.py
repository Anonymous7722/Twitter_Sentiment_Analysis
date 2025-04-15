import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import pandas as pd
import numpy as np
import re

dtc_model = pickle.load(open('dtc_model.pkl','rb'))
lr_model = pickle.load(open('lr_model.pkl','rb'))
vector = pickle.load(open('vector.pkl','rb'))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_and_stem(text):
    # Remove URLs, mentions, hashtags, and special characters
    text = re.sub('[^a-zA-Z]',' ',text)
    text = text.lower()
    tokens = text.split(' ')
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def sentiment_analysis(clean_text):
    sentiment_array=[]
    data=vector.transform(np.array([clean_text]))
    sentiment_array.append(lr_model.predict(data)[0])
    sentiment_array.append(dtc_model.predict(data)[0])
    return sentiment_array

st.title('Twitter Sentiment Analysis')


title = st.text_input('Enter your Tweet')

if st.button('Recommend'):
    with st.status("Model Thinking...") as status:
        clean_text = clean_and_stem(title)
        sentiment_analysis_array = sentiment_analysis(clean_text)  # 0 is Logistic Regression 1 is Decision Tree
        status.update(
        label="Complete!", state="complete", expanded=True)

        st.write('Logistic Regression Model: ',sentiment_analysis_array[0])
        st.write('Decision Tree Model: ',sentiment_analysis_array[1])