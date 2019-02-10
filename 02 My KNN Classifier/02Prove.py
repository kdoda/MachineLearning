# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class MyKNeighborsClassifier:
    def __init__(self,k = 1):
        self.k = k
        self.trainSet = []
        self.trainTargets = []
 
    def fit(self, trainSet, trainTargets):
        #normalize
        self.stdScale = preprocessing.StandardScaler().fit(trainSet)
        self.trainSet = self.stdScale.transform(trainSet)
        self.trainTargets = trainTargets
        
    def predict(self, testSet):   
        testSet = self.stdScale.transform(testSet)
        predictions = []
        for testInstance in testSet:
            prediction = self.__getPrediction(testInstance)
            predictions.append(prediction)
        return predictions
    
    def __euclideanDistance(self,trainInstance, testInstance):
        diff = trainInstance - testInstance
        diff_square = diff ** 2
        return np.sum(diff_square)
    
    def __getNeighbors(self, testInstance):
    	distances = []
    	for trainInstance in self.trainSet:
    		distance = self.__euclideanDistance(trainInstance, testInstance)
    		distances.append(distance)
        
    	sortedDistanceIndexes = np.argsort(distances)
    	neighbors = []
    	for index in sortedDistanceIndexes[:self.k]:
    		neighbors.append(self.trainTargets[index])
    	return neighbors
    
    #if there is a tie between neighbors, it breaks it by choosing the class
    #that has one instance closer to testInstance
    def __getPrediction(self, testInstance):
        neighbors = self.__getNeighbors(testInstance)
        maxFrequency = 0
        mostFrequentNeighbor = 0
        frequencies = {}
        for neighbor in neighbors: 
            if (neighbor in frequencies): 
                frequencies[neighbor] += 1
            else:
                frequencies[neighbor] = 1
                
            thisFrequency = frequencies[neighbor]
            if thisFrequency > maxFrequency:
                maxFrequency = thisFrequency
                mostFrequentNeighbor = neighbor
        
        return mostFrequentNeighbor


iris = datasets.load_iris()

#TODO how to randomize
trainSet, testSet, trainTargets, testTarget = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)


# Initialize our classifier
knn = MyKNeighborsClassifier(k=40)

# Train our classifier
knn.fit(trainSet, trainTargets)

# Make predictions
predictedTargets = knn.predict(testSet)

# Evaluate accuracy
print("%.2f" % accuracy_score(testTarget, predictedTargets))


# up to a k of 35 we get 100% accuracy, even the built in scikit gets the same acccuracy up to that k
# then it starts to drop

#test accuracy against the built in KNN

myAccuracies = [] 
builtinAccuracies = []

neighbors = range(1, 100)
for k in neighbors:
    # build the model
    myKNN = MyKNeighborsClassifier(k = k) 
    myKNN.fit(trainSet, trainTargets)
    
    builtinKNN =  KNeighborsClassifier(n_neighbors=k)   
    builtinKNN.fit(trainSet, trainTargets) 
    
    # record accuracy 
    myAccuracy = accuracy_score(testTarget, myKNN.predict(testSet))
    myAccuracies.append(myAccuracy)

    builtinAccuracy = accuracy_score(testTarget, builtinKNN.predict(testSet))
    builtinAccuracies.append(builtinAccuracy)
    
plt.plot(neighbors, myAccuracies, label="My Accuracies")
plt.plot(neighbors, builtinAccuracies, label="BuiltIn Accuracies")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.legend()

#Stretch challnge
#I'm using my knn for breast cancer dataset
from sklearn.datasets import load_breast_cancer
import pandas as pd

#For future reference
def convertToPandasDataSet(sklearnDataSet):
    df = pd.DataFrame(sklearnDataSet.data, columns=sklearnDataSet.feature_names)
    df['target'] = pd.Series(sklearnDataSet.target)
    return df

def handleNonNumericalData(df):
    columns = df.columns.values

    for column in columns:
        textValues = {}
        def convertToInt(val):
            return textValues[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in textValues:
                    textValues[unique] = x
                    x+=1

            df[column] = list(map(convertToInt, df[column]))

    return df


# check the post for what stratify does
# https://stackoverflow.com/questions/34842405/parameter-stratify-from-method-train-test-split-scikit-learn
cancer = load_breast_cancer()
trainSet, testSet, trainTargets, testTarget = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

accuracies = [] 

# try k from 1 to 100
neighbors = range(1, 100)
for k in neighbors:
    # build the model
    knn = MyKNeighborsClassifier(k = k) 
    knn.fit(trainSet, trainTargets)
    
    # record accuracy 
    accuracy = accuracy_score(testTarget, knn.predict(testSet))
    accuracies.append(accuracy)


plt.plot(neighbors, accuracies, label="accuracies")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.legend()
