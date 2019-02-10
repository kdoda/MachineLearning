#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 23:58:17 2019

@author: klevindoda
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import math
import statistics 

#show the whole table in the console
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

################################STRETCH####################################
#For More information on weighted and gaussian function check:
#Sarma, T. Hitendra et al. An improvement to k-nearest neighbor classifier.
#2011 IEEE Recent Advances in Intelligent Computational Systems (2011): 227-231
class MyKNeighborsRegressor:
    def __init__(self,k = 1, weighted=False):
        self.k = k
        self.trainSet = []
        self.trainTargets = []
        self.weighted = weighted
 
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
    
    def __gaussian(self,dist, sigma=1):
        return 1./(math.sqrt(2.*math.pi)*sigma)*math.exp(-dist**2/(2*sigma**2))
    
    
    def __getPrediction(self, testInstance):
        distances = []
        for trainInstance in self.trainSet:
            distance = self.__euclideanDistance(trainInstance, testInstance)
            distances.append(distance)
        
        sortedDistanceIndexes = np.argsort(distances)

        v = 0
        total_weight = 0
        for index in sortedDistanceIndexes[:self.k]:
            if self.weighted:
                weight = self.__gaussian(distances[index])
                v += self.trainTargets[index] * (weight if self.weighted else 1)
                total_weight += weight
            else:
                v += self.trainTargets[index]
                 
        return (v/total_weight if self.weighted else v/self.k)
   
###############################Car Evaluation###############################

# Define the headers since the data does not have any
headers = ["buying", "maint", "doors", "persons", "lug_boot","safety","car_acceptability"]

df = pd.read_csv("./car.csv", header=None, names=headers)

#check for nulls
df.isnull().values.any()


cleanup_nums = {"buying"  : {"low": 1, "med": 2, "high": 3, "vhigh": 4},
                "maint"   : {"low": 1, "med": 2, "high": 3, "vhigh": 4},
                "doors"   : {"5more": 5},
                "persons" : {"more": 5},
                "lug_boot": {"small": 1, "med": 2, "big": 3},
                "safety"  : {"low": 1, "med": 2, "high": 3}}

cols = df.columns.drop('car_acceptability')

#convert all string numbers to int
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

car_target_names = df.iloc[:,-1]

car_targets = car_target_names.map({"unacc" : 1, "acc" : 1, "good" : 2, "vgood" : 3})

#convert dataframe to numpy arrays
df = df.drop(columns = ["car_acceptability"])
car_data = df.values

train_data, test_data, train_targets, test_targets = train_test_split(car_data, car_targets, test_size=0.3, random_state=42)

# Initialize our classifier
knn = KNeighborsClassifier(n_neighbors=7)

# Train our classifier
knn.fit(train_data, train_targets)

# Make predictions
predicted_targets = knn.predict(test_data)

# Evaluate accuracy
print("%.2f" % accuracy_score(test_targets, predicted_targets))

accuracies = [] 

neighbors = range(1, 30)
for k in neighbors:
    # build the model
    knn = KNeighborsClassifier(n_neighbors = k) 
    knn.fit(train_data, train_targets)
    
    predicted_targets = knn.predict(test_data)
    
    # record accuracy 
    accuracy = accuracy_score(test_targets, predicted_targets) 
    accuracies.append(accuracy)
    
plt.plot(neighbors, accuracies, label="Accuracies")
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.legend()



#########################Automobile MPG########################################
pd.read_csv('mpg.csv',delim_whitespace=True)

# Define the headers since the data does not have any
headers = ["mpg", "cylinders", "displacement", "horsepower", "weight",
              "acceleration","model_year","origin","car_name"]

df = pd.read_csv("./mpg.csv", header=None, names=headers, 
                             delim_whitespace=True, na_values="?")

#how many are nulls?
df.isnull().values.ravel().sum()

#just 6 out of 398, so it's safe to drop them
df.dropna(inplace=True)

#check the shape to make sure we dropped
df.shape

#car_names doesn't really have a meaning if you replace the data
#but it would have meaning since cars in the same brand are similiar
#brand is the first substring of the car_names column, so I can use
#that to create a new column for each brand name using Custom Binary Encoding
#the dimensionality grows ask BR Burton for this 

#Uncomment for one hot encoding
#df["car_name"] = df["car_name"].str.split(' ').str[0]
#car_brands = df.car_name.unique()

#df = pd.get_dummies(df, columns=["car_name"])

mpg_targets = df['mpg'].values
df = df.drop(columns = ["mpg", "car_name"])
mpg_data = df.values

train_data, test_data, train_targets, test_targets = train_test_split(mpg_data, mpg_targets, test_size=0.3, random_state=42)

# Initialize our classifier
knn = KNeighborsRegressor(n_neighbors=7)

# Train our classifier
knn.fit(train_data, train_targets)

# Make predictions
predicted_targets = knn.predict(test_data)

# Evaluate accuracy
print("%.2f" % statistics.mean(abs(predicted_targets - test_targets)))

# =============================================================================
# myAccuracies = [] 
# builtinAccuracies = []
# 
# neighbors = range(1, 30)
# for k in neighbors:
#     # build the model
#     myKNN = MyKNeighborsRegressor(k = k) 
#     myKNN.fit(train_data, train_targets)
#     
#     builtinKNN =  KNeighborsRegressor(n_neighbors=k)   
#     builtinKNN.fit(train_data, train_targets) 
#     
#     # record accuracy 
#     myAccuracy = statistics.mean(abs(test_targets - myKNN.predict(test_data))) 
#     myAccuracies.append(myAccuracy)
# 
#     builtinAccuracy = statistics.mean(abs(test_targets - builtinKNN.predict(test_data))) 
#     builtinAccuracies.append(builtinAccuracy)
#     
# plt.plot(neighbors, myAccuracies, label="My Accuracies")
# plt.plot(neighbors, builtinAccuracies, label="BuiltIn Accuracies")
# plt.ylabel("Accuracy")
# plt.xlabel("k")
# plt.legend()
# 
# =============================================================================
#########################Student G3########################################
df = pd.read_csv("./student-mat.csv",delimiter = ";" )

#drop unrealted columns
df = df.drop(columns = ["address","sex","famsize","reason","guardian","Mjob","Fjob",
                        "schoolsup","famsup","romantic"])
#check for nulls
cleanup_nums = {"paid"       : {"no": 0, "yes": 1},
                "activities" : {"no": 0, "yes": 1},
                "nursery"    : {"no": 0, "yes": 1},
                "higher"     : {"no": 0, "yes": 1},
                "internet"   : {"no": 0, "yes": 1},
                "school"     : {"GP": 0, "MS" : 1},
                "Pstatus"    : {"T" : 0, "A"  : 1}
               }

df.replace(cleanup_nums, inplace=True)

student_targets = df['G3'].values
df = df.drop(columns = ["G3"])
student_data = df.values

train_data, test_data, train_targets, test_targets = train_test_split(student_data, student_targets, test_size=0.3, random_state=42)

# Initialize our classifier
knn = KNeighborsRegressor(n_neighbors=7)

# Train our classifier
knn.fit(train_data, train_targets)

# Make predictions
predicted_targets = knn.predict(test_data)

# Evaluate accuracy
print("%.2f" % statistics.mean(abs(predicted_targets - test_targets)))

# =============================================================================
# myAccuracies = [] 
# builtinAccuracies = []
# 
# neighbors = range(1, 30)
# for k in neighbors:
#     # build the model
#     myKNN = MyKNeighborsRegressor(k = k) 
#     myKNN.fit(train_data, train_targets)
#     
#     builtinKNN =  KNeighborsRegressor(n_neighbors=k)   
#     builtinKNN.fit(train_data, train_targets) 
#     
#     # record accuracy 
#     myAccuracy = statistics.mean(abs(test_targets - myKNN.predict(test_data))) 
#     myAccuracies.append(myAccuracy)
# 
#     builtinAccuracy = statistics.mean(abs(test_targets - builtinKNN.predict(test_data))) 
#     builtinAccuracies.append(builtinAccuracy)
#     
# plt.plot(neighbors, myAccuracies, label="My Accuracies")
# plt.plot(neighbors, builtinAccuracies, label="BuiltIn Accuracies")
# plt.ylabel("Accuracy")
# plt.xlabel("k")
# plt.legend()
# 
# =============================================================================
