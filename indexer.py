import pandas as pd
import numpy as np
import string
import math
import unicodedata as ud

# Function for data preprocessing
def preprocessing(data):

    # Make all letters lowercase
    data["speech"] = data["speech"].str.lower()

    # Remove punctuation
    data["speech"] = data["speech"].str.translate(str.maketrans('', '', string.punctuation))

    #Remove diacritics
    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
    for i in range (0, len(data["speech"])):
        s = ud.normalize('NFD',data["speech"][i]).translate(d)
        data["speech"][i] = s

    # Remove stopwords
    f = open("stopwords.txt", "r")
    stopwords = f.read()
    for i, row in data.iterrows():
        data.at[i,'speech'] = ' '.join([word for word in data.at[i,'speech'].split() if word not in stopwords])
    
    return data
    

# Function to calculate the normalized TF of a word in a document
def termFrequency(term,all_words,max_frequency):
  
  return all_words.count(term) / float(max_frequency)


# Function to calculate TF for every term in every document
def TF_Process(data):
    globalDict = {}     # Global dictionary where:
                            # key = term
                            # value = documents where term can be found

    documentsTF = []    # List that will store dictionaries where:
                            # List index = document id
                                # key = term
                                # value = TF in the particular document

    for i, row in data.iterrows():  #For each document
        speech = data.at[i,'speech']
        
        # Take number of all non-unique words in a document
        all_words = speech.split()

        TF={}   # Dictionary where:
                    # key = term, 
                    # value = TF in the document 
        
        max_frequency = 0
        for term in all_words:

            # update global dictionary with all the documents where a term belongs
            if not term in globalDict:
                globalDict[term] = [i]
            elif not term in TF:
                doc_list = globalDict.get(term)
                doc_list.append(i)
                globalDict[term] = doc_list

            #Calculate maximum frequency of a term in the document
            if(max_frequency <= all_words.count(term)):
                max_frequency = all_words.count(term)

        # calculate the TF for each term
        for term in all_words:    
            TF[term] = termFrequency(term,all_words,max_frequency)
        
        # Update the document TF dictionaries list
        documentsTF.append(TF)
        
    return globalDict, documentsTF


# Function to calculate IDF for every term
def IDF_Process(globalDict,total_docs):
    dictionaryIDF = globalDict.copy()
    for i in dictionaryIDF.keys():
        dictionaryIDF[i] = 1 + math.log(float(total_docs/len(dictionaryIDF[i])))
    return dictionaryIDF