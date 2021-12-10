# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 23:49:25 2021

@author: Mayank 
"""
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random

#%%
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#%%
#cleanup functions definition

def clean_up_sentence(sentence):             #function to convert sentence to lemma words
    sent_words = nltk.word_tokenize(sentence)
    sent_words = [lemmatizer.lemmatize(word.lower()) for word in sent_words]
    return sent_words

def bow(sentence, words, show_details=True):      #bag of words of the sentence
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

#%%
def predict_class(sentence, model):                      # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)           # sort by strength of probability
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#%%
def getResponse(ints, intents_json):     #to get a random response from the list of intents
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

#%%
#developing the gui