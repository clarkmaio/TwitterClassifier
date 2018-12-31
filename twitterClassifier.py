# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:18:56 2018

@author: C320048
"""

import sklearn as sk
import pandas as pd
import numpy as np
import os
import random
import re
import sys

def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))



from sklearn.feature_extraction.text import TfidfVectorizer

def splitData(data, seed = 0, learningSetPerc = 0.7 ):
    N = len(data)
    learning_N = int(N*learningSetPerc)
    
    random.seed(seed)
    randIndex = range(N)
    random.shuffle(randIndex)
    
    shuffleData = data.loc[randIndex].reset_index(drop = True)
    return [ shuffleData.loc[range(learning_N)].reset_index(drop = True), shuffleData.loc[range(learning_N, N)].reset_index(drop = True) ]

def splitPositiveNegative(data):
    positive_data = data[ data["Sentiment"] == 1 ]
    negative_data = data[ data["Sentiment"] == 0 ]    
    return [positive_data, negative_data]


def countSentence(w, data, withRep = False):
    if withRep:
        return sum( [ re.sub("[^\w]", " ", x).lower().split().count(w) for x in data["SentimentText"] ] )
    return sum([w in x for x in data["SentimentText"]])


def LaplaceSmoothing( w, sentimentdata, sub_TotWords, TotWords):   
    return ( countSentence(w, sentimentdata, withRep = True)+1 )/float( sub_TotWords + TotWords )

def idf(w, data):
    return np.log( len(data)/float(countSentence(w, data)) )

def tf_idf(w, data, sentimentdata, sub_TotWords, TotWords):
    return idf(w, data)*LaplaceSmoothing(w, sentimentdata, sub_TotWords, TotWords)



def evaluateSentence(s, data, pos_learSet, neg_learSet, TotWords, pos_TotWords, neg_TotWords):
    wordList = re.sub("[^\w]", " ", s).split()
    pos_prob = np.prod([ tf_idf(w, data, pos_learSet, pos_TotWords, TotWords) for w in wordList])
    neg_prob = np.prod([ tf_idf(w, data, neg_learSet, neg_TotWords, TotWords) for w in wordList])
    
    if pos_prob > neg_prob:
        return 1
    else:
        return 0
    

def totalVocabulary(data):
    f = lambda x,y: x+y
    voc = reduce(f , [ re.sub("[^\w]", " ", x).split() for x in data["SentimentText"]])
    return  list(set([x.lower() for x in voc]))
        
def totalWordNum_withRep(data):
    voc = sum([ len( re.sub("[^\w]", " ", x).split() ) for x in data["SentimentText"]])
    return  voc


os.chdir(get_script_path())

data = pd.read_csv("Sentiment Analysis Dataset.csv")
data = data.loc[:,  ["Sentiment", "SentimentText"] ]


[learSet, testSet] = splitData(data,  learningSetPerc=0.1)

[pos_learSet, neg_learSet] = splitPositiveNegative(learSet)


pos_TotWords = totalWordNum_withRep(pos_learSet)
neg_TotWords = totalWordNum_withRep(neg_learSet)
TotWords = totalWordNum_withRep(learSet)

minitestSet = testSet.loc[range(100)]
minitestSet["Result"] = [ evaluateSentence(s, data, pos_learSet, neg_learSet, TotWords, pos_TotWords, neg_TotWords) for s in minitestSet["SentimentText"] ]
Accuracy = sum( [x == y for (x, y) in zip(minitestSet["Sentiment"], minitestSet["Result"])] )/float(len(minitestSet))




















