import main
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")


tf_idf = None
xg_tfidf  = None

def load_model():
    with open('../Models/TFIDF/tf_idf.pkl', 'rb') as f:
        global tf_idf 
        tf_idf = joblib.load(f) 
    
    with open('../Models/TFIDF/xg_tf_idf.pkl', 'rb') as f:
        global xg_tfidf 
        xg_tfidf = joblib.load(f) 
    
    print(" IF-TDF models successfully loeded")

def query_point_creator_ifidf(q1,q2):
    
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
    q1_idf = tf_idf.transform([q1]).toarray()
    # print(q1_idf.shape)
    
    # bow feature for q2
    q2_idf = tf_idf.transform([q2]).toarray()
    # print(q2_idf.shape)

    other = np.array(input_query).reshape(1,22)
    # print(other.shape)
    
    input = np.hstack((other, q1_idf, q2_idf))

   

    return xg_tfidf.predict(input)
