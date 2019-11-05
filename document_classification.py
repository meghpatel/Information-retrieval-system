
stopwords_path  = '/stopwords.txt'
document_path = 'bbcsport/cricket/001.txt'
preprocesssed_file_path = '/preprocessed.txt'

#using nltk library
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import re
import os
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
import numpy as np


unique_words = set()
count_doc = 0
document_data = []
for folder in os.listdir('bbcsportprocessed/'):
  if not os.path.isdir(os.path.join('bbcsportprocessed/',folder)):
      continue
  for file in os.listdir(os.path.join('bbcsportprocessed/',folder)):
    document_path = os.path.join(os.path.join('bbcsportprocessed/',folder),file)
    count_doc += 1
    print (document_path)
    for x in open(document_path,'r').read().split(' '):
      unique_words.add(x)
unique_words = sorted(list(unique_words))

word_ind = dict(list(zip(unique_words,range(len(unique_words)))))
count = np.zeros((count_doc,len(unique_words))).astype('float32')
doc_num = 0
label = []

for folder in os.listdir('bbcsportprocessed/'):
  if not os.path.isdir(os.path.join('bbcsportprocessed/',folder)):
      continue

  for file in os.listdir(os.path.join('bbcsportprocessed/',folder)):
    document_path = os.path.join(os.path.join('bbcsportprocessed/',folder),file)
    document_data.append(open(document_path,'r').read())

    for x in open(document_path,'r').read().split(' '):
      count[doc_num,word_ind[x]] += 1
    doc_num += 1
    label.append(folder)
total_count_of_word = np.sum(count,axis=1)

for row in range(count.shape[0]):
  for col in range(count.shape[1]):
    count[row][col]=count[row][col]/total_count_of_word[row]

idf = np.log(doc_num/np.sum((count>0).astype(int),axis=0))

tfidf = np.array(list(map(lambda x:x*idf,count)))

#first 20  max
import matplotlib.pyplot as plt
data = list(zip(*sorted(list(enumerate(total_count_of_word)),key = lambda x:-x[1])))
num = 20
plt.bar(range(num),data[1][:num],tick_label=list(map(lambda x:unique_words[x],data[0][:num])))
plt.xticks(rotation='vertical')


boolean = np.array(count>0).astype('int')

print(boolean)

print(tfidf)

print(count)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

#for boolean
X_train, X_test, Y_train, Y_test = train_test_split(boolean,label,test_size=0.2)
mnb_boolean_clf = MultinomialNB()
mnb_boolean_clf.fit(X_train,Y_train)
Y_predict = mnb_boolean_clf.predict(X_test)
print("Accuracy for boolean is {}".format(accuracy_score(Y_test,Y_predict)))

#for count
X_train, X_test, Y_train, Y_test = train_test_split(count,label,test_size=0.2)
mnb_count_clf = MultinomialNB()
mnb_count_clf.fit(X_train,Y_train)
Y_predict = mnb_count_clf.predict(X_test)
# print("Accuracy for multinomial tf is {}".format(accuracy_score(Y_test,Y_predict)))

#for tfidf
X_train, X_test, Y_train, Y_test = train_test_split(tfidf,label,test_size=0.2)
mnb_tfidf_clf = MultinomialNB()
mnb_tfidf_clf.fit(X_train,Y_train)
Y_predict = mnb_tfidf_clf.predict(X_test)
print("Accuracy for tfidf multinomial {}".format(accuracy_score(Y_test,Y_predict)))

from sklearn.naive_bayes import GaussianNB

#for boolean
X_train, X_test, Y_train, Y_test = train_test_split(boolean,label,test_size=0.2)
gnb_boolean_clf = GaussianNB()
gnb_boolean_clf.fit(X_train,Y_train)
Y_predict = gnb_boolean_clf.predict(X_test)
print("Accuracy for gaussian NB boolean {}".format(accuracy_score(Y_test,Y_predict)))

#for count
X_train, X_test, Y_train, Y_test = train_test_split(count,label,test_size=0.2)
gnb_count_clf = GaussianNB()
gnb_count_clf.fit(X_train,Y_train)
Y_predict = gnb_count_clf.predict(X_test)
print("Accuracy for gaussian NB tf {}".format(accuracy_score(Y_test,Y_predict)))

#for tfidf
X_train, X_test, Y_train, Y_test = train_test_split(tfidf,label,test_size=0.2)
gnb_tfidf_clf = GaussianNB()
gnb_tfidf_clf.fit(X_train,Y_train)
Y_predict = gnb_tfidf_clf.predict(X_test)
print("Accuracy TFIDF: Guassian {}".format(accuracy_score(Y_test,Y_predict)))

#SVD
from sklearn.decomposition import TruncatedSVD
print (type(tfidf))
term_doc = tfidf.transpose()

# u, s, vh = np.linalg.svd(term_doc, full_matrices=True)
# print ("Numpy SVD")
# print (u.shape)
# print (s.shape)
# print (vh.shape)
# print (term_doc.shape)
# svd = TruncatedSVD(n_components=100)
# tfidf_new = svd.fit_transform(term_doc)  
print (tfidf.shape)
X_train, X_test, Y_train, Y_test = train_test_split(tfidf,label,test_size=0.2,random_state = 10, stratify=label)
svd = TruncatedSVD(n_components=200)
X_train_new = svd.fit_transform(X_train)  

print (X_train_new.shape)
clf = GaussianNB()
clf.fit(X_train_new,Y_train)

X_test_new = svd.transform(X_test)
Y_predict = clf.predict(X_test_new)
print("Accuracy TFIDF SVD GUASSIAN {}".format(accuracy_score(Y_test,Y_predict)))

# Kmeans
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(X_train)
# print (kmeans.labels_)
output = kmeans.predict(X_test)
# print ("K means without SVD")
# print (output[:100])
# print (Y_train[:100])
# Kmeans SVD
from sklearn.cluster import KMeans
kmeans1 = KMeans(n_clusters=5, random_state=0).fit(X_train_new)
# print ("K means with SVD")
# print (kmeans1.labels_)
output1 = kmeans1.predict(X_test_new)
# print (output1[:100])
# print ("Y_Test: ")
# print (Y_test[:100])

# Genetic Algorithms
"""
[(),(),(),(),()]

FF 

TFIDF
1) Form cluster with random centroids generated
2) Find intra-cluster and inter-cluster distance. 
  2a) Intra-cluster = Max(distance(centroid, any element in that cluster))
  2b) Inter-cluster
3) Fitness function = Inter/Intra
"""