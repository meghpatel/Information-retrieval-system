import string 
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from textblob import TextBlob
import os

stopwords = set(open('stopwords.txt','r').read().split('\n'))

punctuations = [char for char in string.punctuation]

directory_in_str = 'bbcsport/'
final_directory_in_str = 'bbcsportprocessed/'

def removeStopwords(data):
    # print (data)
    newdata = data.lower().replace('\n',' ').split(' ')
    # print (type(newdata))
    newwords = []
    for i,word in enumerate(newdata):
        if word not in stopwords:
            if len(word) != 0 and word[0] in punctuations:
                word = word[1:]
            if len(word) != 0 and word[-1] in punctuations:
                word = word[:-1]
            newwords.append(word)
    return ' '.join(newwords)

def savetopreprocess(folder, filename, content):
    file = final_directory_in_str + folder + '/' + str(filename)
    fname = final_directory_in_str + folder
    exists = os.path.isdir(fname)

    # print (exists)
    if not exists:
        # os.mkdir(fname)        
        exists1 = os.path.isdir(final_directory_in_str)

        if not exists1:
            os.mkdir(final_directory_in_str)
        os.mkdir(fname)

    try:
        with open(file, 'w') as f:
            f.write(content)
        return True
    except Exception as e:
        print (e)
        return False


def preprocess(folder, filename):
    file = directory_in_str + folder + '/' + str(filename)
    try:
        with open(file) as f:
            data = f.read()
            d1 = removeStopwords(data)
            # print (d1)
            state = savetopreprocess(folder,filename, d1)
            return state
    except Exception as e:
        print (e)
        return False


# with open('001.txt','r') as f:
#     data = f.read()
#     d1 = removeStopwords(data)
#     # print (d1)


directory = os.fsencode(directory_in_str)

for folders in os.listdir(directory):
    print (folders)
    try:
        folder = directory_in_str + str(folders.decode('utf-8'))
        for files in os.listdir(folder):
            file = os.fsdecode(files)
            state = preprocess(str(folders.decode('utf-8')),file)
            print (file)
    except Exception as e:
        print (e)
        continue
            # print (state)
