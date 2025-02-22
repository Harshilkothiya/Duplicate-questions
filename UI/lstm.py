import main
import numpy as np
import joblib
import warnings
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")


tokenizer = None
lstm  = None

def load_model():
    with open('../Models/LSTM/tokenizer_lstm.pkl', 'rb') as f:
        global tokenizer 
        tokenizer = joblib.load(f) 
    
    with open('../Models/LSTM/model_lstm.pkl', 'rb') as f:
        global lstm 
        lstm = joblib.load(f) 
    
    print(" LSTM models successfully loeded")

def query_point_creator_lstm(q1,q2):
    
    input_query = []
    
    # preprocess
    q1 = main.preprocess(q1)
    q2 = main.preprocess(q2)
    
    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(main.test_common_words(q1,q2))
    input_query.append(main.test_total_words(q1,q2))
    input_query.append(round(main.test_common_words(q1,q2)/main.test_total_words(q1,q2),2))
    
    # fetch token features
    token_features = main.test_fetch_token_features(q1,q2)
    input_query.extend(token_features)
    
    # fetch length based features
    length_features = main.test_fetch_length_features(q1,q2)
    input_query.extend(length_features)
    
    # fetch fuzzy features
    fuzzy_features = main.test_fetch_fuzzy_features(q1,q2)
    input_query.extend(fuzzy_features)
    
    # bow feature for q1
    q1_lstm = tokenizer.texts_to_sequences([q1])
    q1_lstm = pad_sequences(q1_lstm, maxlen=50)
    # print(q1_lstm.shape)
    
    # bow feature for q2
    q2_lstm = tokenizer.texts_to_sequences([q2])
    q2_lstm = pad_sequences(q2_lstm, maxlen=50)
    # print(q2_lstm.shape)

    other = np.array(input_query).reshape(1,22)
    # print(other.shape)
    
    input = np.hstack((other, q1_lstm, q2_lstm))

    if(lstm.predict(input)) > 0.5 :
        return 0
    else:
        return 1
