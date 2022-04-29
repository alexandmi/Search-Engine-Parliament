import pandas as pd
import numpy as np
import string
import math
import unicodedata as ud
from indexer import preprocessing, termFrequency, TF_Process, IDF_Process


# Function for query preprocessing
def query_Preprocessing(query):
    
    # Make all letters lowercase
    query = query.lower()

    # Remove punctuation
    query = query.translate(str.maketrans('', '', string.punctuation))

    # Remove diacritics
    query = query.split()
    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
    for i in range (0, len(query)):
        s = ud.normalize('NFD',query[i]).translate(d)
        query[i] = s

    # Remove stopwords
    f = open("stopwords.txt", "r")
    stopwords = f.read()
    query = ' '.join([word for word in query if word not in stopwords])
    
    return query.split()


# Function to calculate TF for every term in the query
def query_TF_Process(query, data, globalDict):
    
    queryGlobalDict = globalDict.copy()
    
    TF={}   # Dictionary where:
                # key = term, 
                # value = TF in the query
    
    max_frequency = 0
    for term in query:
    
        # Update the query global dictionary
        if not term in queryGlobalDict:
            queryGlobalDict[term] = ["query"]
        elif not term in TF:
            doc_list = queryGlobalDict.get(term)
            doc_list.append("query")
            queryGlobalDict[term] = doc_list

        #Calculate maximum frequency of a term in the query
        if(max_frequency <= query.count(term)):
            max_frequency = query.count(term)

    # calculate the TF for each term
    for term in query:
        TF[term] = termFrequency(term,query,max_frequency)
    
    return queryGlobalDict, TF


# Function to calculate IDF for every term
def query_IDF_Process(query, queryGlobalDict, dictionaryIDF, total_docs):
    queryDictionaryIDF = dictionaryIDF.copy()
    for i in query:
        queryDictionaryIDF[i] = 1 + math.log(float(total_docs/len(queryGlobalDict[i])))
    
    return queryDictionaryIDF



# Function to create the tf-idf vector of a document 
def create_vector(single_doc, queryDictionaryIDF, queryGlobalDict):
    doc_tfidf = {}
    
    # Every weight is initially 0
    for term in queryGlobalDict.keys():
        doc_tfidf[term] = 0
    
    for term in queryGlobalDict.keys():
        if term in single_doc.keys():
            doc_tfidf[term] = single_doc[term]* queryDictionaryIDF[term]

    return doc_tfidf



# Function to perform cosine similarity
def cosine_formula(query,doc):
    
    dot_product = np.dot(query,doc)
    query_norm = np.linalg.norm(query)
    doc_norm = np.linalg.norm(doc)

    return (dot_product/query_norm*doc_norm)



# This is the function that builds the similarity matrix between the query and the documents
def cosine_similarity(query_vector_list,tfidf_list):
    similarity_matrix = [cosine_formula(query_vector_list,list(doc.values())) for doc in tfidf_list]
    
    return similarity_matrix



# This is the main query search function
def query_search(query):
    
    data = pd.read_csv("parliament.csv")
    
    # Preprocess data
    data = preprocessing(data)

    # globalDict: dictionary with all the terms (keys) and the documents where they can be found (values)
    # documentsTF: list of dictionaries with terms and their frequency in a document
    globalDict, documentsTF = TF_Process(data)

    # dictionaryIDF: dictionary with all the terms (keys) and their respective IDF (values)
    dictionaryIDF = IDF_Process(globalDict,data.shape[0]) 

    # Preprocess query
    query = query_Preprocessing(query)
    # Same as the dicionaries above, with the addition of the query terms in them
    queryGlobalDict, queryTF = query_TF_Process(query,data,globalDict)
    queryDictionaryIDF = query_IDF_Process(query, queryGlobalDict, dictionaryIDF, data.shape[0])


    #Fill the tf of terms not included in a document with zeroes
    for i in queryGlobalDict.keys():
        for j in range(len(documentsTF)):
            if not i in documentsTF[j]:
                documentsTF[j][i]=0

    # Create tf-idf vectors for all the documents and put them on a list
    tfidf_list = []
    for single_doc in documentsTF:
        tfidf_list.append(create_vector(single_doc, queryDictionaryIDF, queryGlobalDict))


    # Create a dictionary for all terms tf-idf 
    queryTFIDF={}
    for term in queryGlobalDict.keys():
        queryTFIDF[term] = 0
        
    for i in query:
        queryTFIDF[i] = queryDictionaryIDF[i]*queryTF[i]


    # Turn the query TFIDF dictionary to a list
    query_vector_list = list(queryTFIDF.values())
    # Compute the similarity matrix between our query and our documents
    similarity_matrix = cosine_similarity(query_vector_list,tfidf_list)
    query_data = data.copy()
    #query_data['TFIDF'] = tfidf_list  # you can put the tfidf data in your final results if you want
    query_data['Score'] = np.array(similarity_matrix)
    query_data.sort_values(by=['Score'], inplace=True, ascending=False)
    return query_data, queryTFIDF