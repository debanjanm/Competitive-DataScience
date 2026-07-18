###############################################################################
###############################################################################
# importing required libraries
import pandas as pd
import numpy as np
import os

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
# import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from helpers import *

from config import TRAIN,TEST
train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)

def preprocess():

    data = train

    #Remove irrelevant features 
    columns = ['Id']
    data.drop(columns, inplace=True, axis=1)

    # Preprocessing
    data['Review'] = data['Review'].str.lower()
    data['Review'] = data['Review'].apply(decontract)
    data['Review'] = data['Review'].apply(cleanPunc)
    data['Review'] = data['Review'].apply(keepAlpha)
    data['Review'] = data['Review'].apply(removeStopWords)
    data['Review'] = data['Review'].apply(stemming)

    #Split
    X_train, X_test, y_train, y_test = train_test_split(data['Review'], 
                                                        data[data.columns[1:]], 
                                                        test_size=0.3, 
                                                        random_state=43, 
                                                        shuffle=True)


    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vectorizer.fit(X_train)

    features_train_transformed = vectorizer.transform(X_train)
    features_test_transformed  = vectorizer.transform(X_test)
    labels_train = y_train
    labels_test  = y_test
    return data,features_train_transformed, features_test_transformed, labels_train, labels_test





