#Music Genre Classification
#libraries
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from tempfile import TemporaryFile
import os
import math
import pickle
import random
import operator

#Mahalanobis Distance
def distance(inst1, inst2):
     dist = 0
     mm1 = inst1[0] #mean matrix
     cm1 = inst1[1] #covariance matrix
     mm2 = inst2[0]
     cm2 = inst2[1]
     dist = np.sqrt(np.dot((mm2-mm1).transpose(), np.dot(np.linalg.inv(cm2), (mm2-mm1))))
     return dist

#get distance between feature vectors and find neighbors
def findNeighbors(trainSet, instance, k):
    distances = []
    for x in range(len(trainSet)):
        dist = distance(trainSet[x], instance) + distance(instance,trainSet[x])
        distances.append((trainSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#identify the nearest class 
def nearestclass(neighbors):
    classVote = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
        else:
            classVote[response] = 1
            
    sorter = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
    return sorter[0][0]

#feature extraction using mfcc (Mel-frequency cepstral coefficients)

directory = 'C:/Users/sinha/OneDrive/Desktop/Khushi/6th sem/cognitive analytics/Project/archive/Data/genres_original'
f = open("mydataset.dat", "wb")
i = 0
for folder in os.listdir(directory):
    print(folder)
    i += 1
    if i == 11:
        break
    for file in os.listdir(directory+"/"+folder):
        try:
            (rate, sig) = wav.read(directory+"/"+folder+"/"+file)
            mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy=False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
        except Exception as e:
            print("Got an exception: ", e, 'in folder: ', folder, ' filename: ', file)
f.close()

#spliting dataset into train- test set
dataset = []

def loadDataset(filename, split, trset, teset):
    with open('mydataset.dat','rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break
    for x in range(len(dataset)):
        if random.random() < split:
            trset.append(dataset[x])
        else:
            teset.append(dataset[x])

trainSet = []
testSet = []
loadDataset('mydataset.dat', 0.68, trainSet, testSet)

# Make the prediction using KNN(K nearest Neighbors)
length = len(testSet)
predictions = []
for x in range(length):
    predictions.append(nearestclass(findNeighbors(trainSet, testSet[x], 5)))

def getAccuracy(testSet, prediction):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
    return 1.0 * correct / len(testSet)
accuracy1 = getAccuracy(testSet, predictions)
print(accuracy1)

from collections import defaultdict
results = defaultdict(int)

directory = "C:/Users/sinha/OneDrive/Desktop/Khushi/6th sem/cognitive analytics/Project/archive/Data/genres_original"

i = 1
for folder in os.listdir(directory):
    results[i] = folder
    i += 1

pred= nearestclass(findNeighbors(dataset, feature, 5))
print(results[pred])

