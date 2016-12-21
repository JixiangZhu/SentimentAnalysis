'''
Created on Dec 20, 2016

@author: jixiang
'''
from __future__ import division, print_function, absolute_import
import tflearn
import datasets
from datasets.data_utils import to_categorical, pad_sequences
# from tflearn.data_utils import to_categorical, pad_sequences
# from tflearn.datasets import imdb

# IMDB Dataset loading
import sys
import re
import os
import glob

pos_train_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/train/pos/*.txt')
neg_train_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/train/neg/*.txt')
pos_test_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/test/pos/*.txt')
neg_test_files = glob.glob('/media/jixiang/Local Disk/KTH/Deep Learning/project/aclImdb/test/neg/*.txt')

trainX = []
testX = []
trainY = []
testY = []

for fileName in pos_train_files:
    fin = open( fileName, "r" )
    review = re.split('\W+', fin.read())
    trainX.append(review)
    trainY.append(1)
    fin.close() # closes file
    
for fileName in neg_train_files:
    fin = open( fileName, "r" )
    review = re.split('\W+', fin.read())
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

# train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
#                                 valid_portion=0.1)
# trainX, trainY = train
# testX, testY = test

# Data preprocessing
# Sequence padding
print (trainX[1])
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
batch_size=32)