# -*- coding: utf-8 -*-
"""irs_preprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fI_6xsLK5aWJAJZqlXm8yUYMUWgaw0x9
"""

# from google.colab import drive
# drive.mount('/content/gdrive')

# cd "gdrive/My Drive/Sem 7/IRS"

stopwords = set(open('stopwords.txt','r').read().split('\n'))

import string 
punctuations = [char for char in string.punctuation]

def removeStopwords(data):
    # print (data)
    newdata = data.lower().replace('\n',' ').split(' ')
    print (type(newdata))
    newwords = []
    for i,word in enumerate(newdata):
        if word not in stopwords:
            if len(word) != 0 and word[0] in punctuations:
                word = word[1:]
            if len(word) != 0 and word[-1] in punctuations:
                word = word[:-1]
            newwords.append(word)
    

    # print (new)
    # for stword in stopwords:
    #     # print ("Inside If")
    #     newdata.replace(stword,'')
    return ' '.join(newwords)

with open('drugdiscovery.txt','r') as f:
    data = f.read()
    # print (data)
    d1 = removeStopwords(data)
    print (d1)
    # if data == 'Modern':
    #     print ("Yes")

"""# Using NLTK

## Tokenize
"""

from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

input_str = word_tokenize(open('drugdiscovery.txt','r').read())
print(input_str)

"""### Removing punctuations"""

import string 
table = str.maketrans('','',string.punctuation)
clean = [w.translate(table) for w in input_str]
words = [w for w in clean if w.isalpha()]

print (words)

"""## Removing Stop Words"""

nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
new_words = [w.lower() for w in words if not w in stop_words]

"""## Stemming"""

from nltk.stem import PorterStemmer

stemmer= PorterStemmer()

stemmmed_words = []
for word in new_words:
    stemmmed_words.append(stemmer.stem(word))

print (stemmmed_words)

from nltk.stem.lancaster import LancasterStemmer

stemmer1 = LancasterStemmer()

stemmmed_words1 = []
for word in words:
    stemmmed_words1.append(stemmer1.stem(word))

print (stemmmed_words1)

"""## POS Tagging"""

from textblob import TextBlob
result = TextBlob(input_str)
print(result.tags)

vocab = d1.split(' ')

print (len(vocab))

"""# TF-IDF Model"""

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

import os
for i, filename in enumerate(os.listdir('bbcsport')):
    if filename.endswith(".txt"): 
        print(os.path.join('bbcsport', filename))
        file = os.path.join('bbcsport', filename)
        with open(file,'r') as f:
            data = f.read()
            # print (data)
            d1 = removeStopwords(data)
            fname = str(i) +'pro.txt'
            f = open(fname,'w')
            f.write(d1)
            print (d1)
        continue
    else:
        continue

vocab = []
vocab1 = []
for i, filename in enumerate(os.listdir('bbc_processed')):
    if filename.endswith(".txt"): 
#         print(os.path.join('bbc_processed', filename))
        file = os.path.join('bbc_processed', filename)
        with open(file,'r') as f:
            data = f.read()
            words = data.split(' ')
            for word in words:
                vocab1.append(word)
                if word not in vocab:
                    vocab.append(word)
#         print (vocab)
        continue
    else:
        continue

len(vocab)

import pandas as pd
