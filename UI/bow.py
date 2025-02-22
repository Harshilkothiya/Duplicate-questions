import main
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")


cv = None
rf_bw  = None

def load_model():
    with open('../Models/BOW/cv_bow.pkl', 'rb') as f:
        global cv 
        cv = joblib.load(f) 
    
    with open('../Models/BOW/bagofword.pkl', 'rb') as f:
        global rf_bw 
        rf_bw = joblib.load(f) 
    
    print("BOW models successfully loeded")

def query_point_creator_bag_of_word(q1,q2):
    
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
    q1_bow = cv.transform([q1]).toarray()
    # print(q1_bow.shape)
    
    # bow feature for q2
    q2_bow = cv.transform([q2]).toarray()
    # print(q2_bow.shape)
    
    input = np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))
    return rf_bw.predict(input)
