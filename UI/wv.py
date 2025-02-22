import main
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but.*")


model = None
rf_wtov  = None

def load_model():
    with open('../Models/WTOV/model_wtov.pkl', 'rb') as f:
        global model 
        model = joblib.load(f) 
    
    with open('../Models/WTOV/wtov.pkl', 'rb') as f:
        global rf_wtov 
        rf_wtov = joblib.load(f) 
    
    print(" WV models successfully loeded")

def sentence_vector(sentence, model):
    
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]  # Ignore OOV words
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)  # Return zero vector if all words are OOV


def query_point_creator_w2v(q1,q2):
    
    input_query = []
    
    # preprocess
    q1 = main.preprocess(q1)
    q2 = main.preprocess(q2)
    
    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(main.test_total_words(q1,q2))
    input_query.append(main.test_common_words(q1,q2))
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
    q1_wv = sentence_vector(q1, model).reshape(1, 300)
    # print(q1_wv.shape)
    
    # bow feature for q2
    q2_wv = sentence_vector(q2, model).reshape(1, 300)
    # print(q2_wv.shape)
    
    input = np.hstack((np.array(input_query).reshape(1,22),q1_wv,q2_wv))
    return rf_wtov.predict(input)
