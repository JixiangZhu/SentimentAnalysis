'''
Created on Dec 20, 2016

@author: jixiang
'''
# import re
# text_file = open("/home/jixiang/workspace/LSTM/dataset/0_3.txt", "r")
# lines = re.split('\W+', text_file.read())
# print lines
import re
import os
import glob
import datasets
from datasets.data_utils import to_categorical, pad_sequences
trainX = []
testX = []
trainY = []
testY = []
pos_train_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/train/pos/*.txt')
neg_train_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/train/neg/*.txt')
pos_test_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/test/pos/*.txt')
neg_test_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/test/neg/*.txt')

for fileName in pos_train_files:
    fin = open( fileName, "r" )
    review = re.split('\W+', fin.read()[:-1])
    trainX.append(review)
    trainY.append(1)
    fin.close() # closes file
    
for fileName in neg_train_files:
    fin = open( fileName, "r" )
    review = re.split('\W+', fin.read()[:-1])
    trainX.append(review)
    trainY.append(0)
    fin.close() # closes file
    
for fileName in pos_test_files:
    fin = open( fileName, "r" )
    review = re.split('\W+', fin.read())
    testX.append(review)
    testY.append(1)
    fin.close() # closes file
    
for fileName in neg_test_files:
    fin = open( fileName, "r" )
    review = re.split('\W+', fin.read())
    testX.append(review)
    testY.append(0)
    fin.close() # closes file
    
# Data preprocessing
# Sequence padding
print (trainX[1])

trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
    
