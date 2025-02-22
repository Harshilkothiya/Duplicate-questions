import streamlit as st
import numpy as np
import time
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pickle
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all messages, 1 = warnings, 2 = errors, 3 = fatal errors
import tensorflow as tf


warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")

#import model
import tfidf
import bow
import wv
import lstm

print("start this file")

# Home page
st.set_page_config(
    page_title="Smart Crop",
    page_icon="logo.webp",
    layout="centered",
)

def main():
    st.header("Check Qustions is Dublicat or Not")
    q1 = st.text_input("Enter First question")
    q2 = st.text_input('Enter Second question')

    if q1 and q2:
        st.subheader("Select Model for Prediction:")
        
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("Predict with TFIDF"):
                output = tfidf.query_point_creator_ifidf(q1, q2)
                if(output == 1):
                    st.warning("Both are Same")
                else:
                    st.success("Both are Different") 


        with col2:
            if st.button("Predict with LSTM"):
                output = lstm.query_point_creator_lstm(q1, q2)
                if(output == 1):
                    st.warning("Both are Same")
                else:
                    st.success("Both are Different") 


        with col3:
            if st.button("Predict with BOW"):
                output = bow.query_point_creator_bag_of_word(q1, q2)
                if(output == 1):
                    st.warning("Both are Same")
                else:
                    st.success("Both are Different") 


        with col4:
            if st.button("Predict with W2V"):
                output = wv.query_point_creator_w2v(q1, q2)
                if(output == 1):
                    st.warning("Both are Same")
                else:
                    st.success("Both are Different") 

    else:
        st.warning("Please enter both questions to proceed!")

    


if __name__ == "__main__":
    tfidf.load_model()
    bow.load_model()
    wv.load_model()
    lstm.load_model()
    main()