# -*- coding: utf-8 -*-
"""
Created on Thur Nov  5 15:58:03 2020

@author: psarikh
"""

# read the data:
    
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from nltk.stem.porter import *
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import re
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
import gensim
from gensim.models import KeyedVectors
import Levenshtein
import numpy as np 


st = stopwords.words('english')
stemmer = PorterStemmer()


def loadDataAsDataFrame(f_path):
    '''
        Given a path, loads a data set and puts it into a dataframe
        - simplified mechanism
    '''
    df = pd.read_csv(f_path)
    return df



def preprocess_text(raw_text):    
    '''        Preprocessing function        
    PROGRAMMING TIP: Always a good idea to have a *master* preprocessing function that reads in a string and returns the       
    preprocessed string after applying a series of functions.    '''    
    
    # Replace/remove username    
    raw_text = re.sub('(@[A-Za-z0-9\_]+)', '', raw_text) #@username_ 
    
    # Remove url
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    raw_text = re.sub(regex, '', raw_text)
    
    # Remove non english characters
    raw_text = re.sub('[^\u0000-\u05C0\u2100-\u214F]', '', raw_text)
    
    #stemming and lowercasing (no stopword removal  
    words = [stemmer.stem(w) for w in raw_text.lower().split()]    
    
    return words #(" ".join(words))


#-----------------------------------------------------------------------------
#TODO : This is related to classifiers and might be modified
def grid_search_hyperparam_space(params, pipeline, folds, training_texts, training_classes):#folds, x_train, y_train, x_validation, y_validation):
        grid_search = GridSearchCV(estimator=pipeline, param_grid=params, refit=True, cv=folds, return_train_score=False, scoring='accuracy',n_jobs=-1)
        grid_search.fit(training_texts, training_classes)
        return grid_search
#-----------------------------------------------------------------------------


if __name__ == '__main__':
    #Load the data
    f_path = './Breast Cancer(Raw_data_2_Classes).csv'
    data = loadDataAsDataFrame(f_path)
    texts = data['Text']
    classes = data['Class']
    ids = data['ID']

    
    #PREPROCESS THE text feature:
    texts_preprocessed = []
    
    print('preprocessing the input data')
    for tr in texts:
        # PREPROCESS THE DATA
        texts_preprocessed.append(preprocess_text(tr))


    print('------------ Part 1 -----------')
    print("20 most similar terms for 'tamoxifen' using word2vec model")
    # Generate a word2vec model
    model = gensim.models.Word2Vec(texts_preprocessed, min_count=10, size=200)
    # Storing and loading models
    model.save('./mymodel')
    new_model = gensim.models.Word2Vec.load('./mymodel')
    # using the model for getting similar words
    print(model.most_similar(positive=['tamoxifen'], topn=20))


    print('------------ Part 2 ------------')
    print('load the model')
    filename = './DSM-language-model-1B-LARGE/trig-vectors-phrase.bin'
    word_vectors = KeyedVectors.load_word2vec_format(filename, binary=True, encoding='utf8', unicode_errors='ignore')
    dic_words={}
    print('get the top 10000 most similar words')
    top_words = word_vectors.most_similar("tamoxifen", topn=10000)
    print('calculate the Levenshtein ratio')
    for term in top_words:
        dic_words[term[0]] = Levenshtein.ratio('tamoxifen', term[0]) # returns 1
    print('sort the similar words based of the value of the levenshtein ratio')
    sorted_dic_words = sorted(dic_words.items(), key=lambda x: x[1], reverse=True)
    print('Top 100 most similar terms for tamoxifen')
    print(np.transpose(sorted_dic_words[:100])[0])
    
    print('similar words to tamoxifen with levenshtein ratio = 0.8')
    ratio = 0.8
    for key, value in sorted_dic_words:
        if value>=ratio:
            print(key)
    
    print('similar words to tamoxifen with levenshtein ratio = 0.75')
    ratio = 0.75
    for key, value in sorted_dic_words:
        if value>=ratio:
            print(key)
            
    print('similar words to tamoxifen with levenshtein ratio = 0.7')
    ratio = 0.7
    for key, value in sorted_dic_words:
        if value>=ratio:
            print(key)
            
    print('similar words to tamoxifen with levenshtein ratio = 0.6')
    ratio = 0.6
    for key, value in sorted_dic_words:
        if value>=ratio:
            print(key)
    
    
# ----------------------------Output of the code: ----------------------------
# preprocessing the input data
# ------------ Part 1 -----------
# 20 most similar terms for 'tamoxifen' using word2vec model
# C:\Users\psarikh\Downloads\NLP_Fall2020\hw 10\Parisa_Sarikhani_HW10.py:97: DeprecationWarning: Call to deprecated `most_similar` (Method will be removed in 4.0.0, use self.wv.most_similar() instead).
#   print(model.most_similar(positive=['tamoxifen'], topn=20))
# [('possibl', 0.9998699426651001), ('anoth', 0.9998480081558228), ('dr', 0.9998313188552856), ('cut', 0.9998194575309753), ('side', 0.9998167157173157), ('diseas', 0.9998133182525635), ('into', 0.999811589717865), ('less', 0.9997965097427368), ('herceptin', 0.9997916221618652), ('condit', 0.9997830390930176), ('call', 0.9997823238372803), ('hard', 0.9997797012329102), ('run', 0.9997789263725281), ('healthi', 0.999777615070343), ('leav', 0.9997775554656982), ('posit', 0.9997773766517639), ('meet', 0.9997769594192505), ('around', 0.9997739791870117), ('tough', 0.9997617602348328), ('right', 0.9997557401657104)]
# ------------ Part 2 ------------
# load the model
# get the top 10000 most similar words
# calculate the Levenshtein ratio
# sort the similar words based of the value of the levenshtein ratio
# Top 100 most similar terms for tamoxifen
# ['tamoxifin' 'tamoxifan' 'tomoxifen' 'tamoxafin' 'raloxifene' 'tamox'
#  'amoxapine' 'amoxil' 'amoxillin' 'tamsulosin' 'amoxicilin' 'amoxcillin'
#  'raloxifine' 'amoxicllin' 'cytoxin' 'lanoxin' 'rutaxin' 'tovaxin' 'amox'
#  'lamotrigene' 'amoxicillan' 'amoxacillin' 'amoxocillin' 'amoxycillin'
#  'amoxicillin' 'atomoxetine' 'eltroxin' 'tagament' 'thyroxin' 'tratment'
#  'bacoflen' 'ampligen' 'baclofen' 'ambiencr' 'treament' 'vitamind'
#  'teatment' 'naproxen' 'treatmen' 'naproxin' 'aloxi' 'taxel'
#  'nitazoxanide' 'acetaminopen' 'moxifloxacin' 'terazosin' 'flutamide'
#  'treaments' 'amiloride' 'remifemin' 'ambien_xr' 'cytotoxin' 'metmorfin'
#  'metaforin' 'pizotifen' 'mammosite' 'thyroxine' 'treatmens' 'vitamind3'
#  'carprofen' 'metafolin' 'clomifene' 'thryoxine' 'betaferon' 'meloxican'
#  'naproxene' 'mag_oxide' 'acetaminophin' 'statin' 'yasmin' 'biaxin'
#  'atavin' 'topamx' 'diamox' 'lutien' 'toxity' 'takinh' 'difene' 'targin'
#  'taxane' 'mirena' 'patien' 'talwin' 'roiben' 'metaformin' 'amlodipine'
#  'ergotamine' 'dobutamine' 'treamtents' 'paroxetine' 'mexiletine'
#  'gentamicin' 'amioderone' 'famotadine' 'famotidine' 'paroxotine'
#  'tacrolinus' 'gentamycin' 'vitamin_k1' 'trreatment']
# similar words to tamoxifen with levenshtein ratio = 0.8
# tamoxifin
# tamoxifan
# tomoxifen
# similar words to tamoxifen with levenshtein ratio = 0.75
# tamoxifin
# tamoxifan
# tomoxifen
# tamoxafin
# similar words to tamoxifen with levenshtein ratio = 0.7
# tamoxifin
# tamoxifan
# tomoxifen
# tamoxafin
# raloxifene
# tamox
# similar words to tamoxifen with levenshtein ratio = 0.6
# tamoxifin
# tamoxifan
# tomoxifen
# tamoxafin
# raloxifene
# tamox
# amoxapine
# amoxil
# amoxillin
# tamsulosin
# amoxicilin
# amoxcillin
# raloxifine
# amoxicllin
# cytoxin
# lanoxin
# rutaxin
# tovaxin
# amox
# lamotrigene
# amoxicillan
# amoxacillin
# amoxocillin
# amoxycillin
# amoxicillin
# atomoxetine