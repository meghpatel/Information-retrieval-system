import string 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from textblob import TextBlob

stopwords = set(open('stopwords.txt','r').read().split('\n'))

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

