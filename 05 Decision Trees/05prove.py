# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 16:04:27 2019

@author: kdoda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:03:15 2019

@author: klevindoda
"""

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

#########################Implementation Option#################################

#Do wee need to store the feature names
#If so it's easy just get the columns from dataset
#we already have the indexes on each node
class MyDecisionTreeClassifier:
    def __init__(self):
        self.data = []
        self.targets = []

    # expecting numpy arrays
    def fit(self, data, targets):
        self.targets = targets
        self.data = data
        feature_indexes = np.arange(0, self.data.shape[1], 1)
        self.tree = self.__make_tree(self.data,self.targets,feature_indexes)   
        
    def predict(self, test_data):  
        predicted_targets = np.array([],dtype=test_data.dtype)
        for data in test_data:
            predicted_target = self.__predict_target(data)
            predicted_targets = np.append(predicted_targets,predicted_target)
        return predicted_targets
            
    def __get_child(self,node):
        return list(node.keys())[0]
    
    def __isLeaf(self,instance):
        return not isinstance(instance, dict)

    #TODO when the data doesn't match Found input variables with inconsistent numbers of samples
    def __predict_target(self, datapoint): 
        node = self.tree
        while not self.__isLeaf(node):
            child = self.__get_child(node)
            node = node[child][datapoint[child]]
        return node
            
    def __calc_entropy(self,p):
        if p!=0:
            return -p * np.log2(p)
        else:
            return 0

    def __calc_averaged_entropy(self,data,classes,featureIndex):
        
        uniqueFeatureValues = set()
        # go through the rows 
        for datapoint in data:
            #set will store dublicates
            uniqueFeatureValues.add(datapoint[featureIndex])
            
        #turn it back to a list
        uniqueFeatureValues = list(uniqueFeatureValues)
        
        averagedFeatureEntropy = 0
        # Find where those values appear in data[feature] 
        # and the corresponding class
        for featureValue in uniqueFeatureValues:
            featureValueEntropy = 0
            
            #classes mapped to this feature value
            newClasses = []
            dataIndex = 0
            for datapoint in data:
                if datapoint[featureIndex]==featureValue: 
                    newClasses.append(classes[dataIndex])
                dataIndex += 1
                 
            # Dictionary that maps classes values to frequency
            classesFrequency = {}
            for classValue in newClasses:
                if classValue in classesFrequency:
                    classesFrequency[classValue] += 1
                else:
                    classesFrequency[classValue] = 1
    
            # how many classes are there in total, so we can calculate the probability
            totalClassValues = sum(classesFrequency.values())
            
            for classValue,frequency in classesFrequency.items():
                featureValueEntropy += self.__calc_entropy(float(frequency)/totalClassValues)
            
            averagedFeatureEntropy += totalClassValues/len(data) * featureValueEntropy
        return averagedFeatureEntropy


    def __max_repeated(self,classes):
        assert len(classes) != 0
    
        seen = set()
        dups = set()
        for x in classes:
            if x in seen:
                dups.add(x)
            seen.add(x)
        return max(dups) if len(dups) != 0 else classes[0]
         
    def __make_tree(self,data,classes,feature_indexes):
        nData = len(data)
        nFeatures = len(feature_indexes)
        
        #No Data left
        if nData==0:
            return ""
        # No features left to test
        elif nFeatures == 0:
            # Have reached an empty branch
            return self.__max_repeated(classes)
        # Only 1 class remains
        elif list(classes).count(classes[0]) == nData:
            return classes[0]
        else:
            # Choose which feature is best
            # TODO put this on a method
            averaged_entropies = np.zeros(nFeatures)
            for featureIndex in range(nFeatures):
                averaged_entropies[featureIndex] = self.__calc_averaged_entropy(data,classes,featureIndex)

            bestFeature = np.argmin(averaged_entropies)
            
            #store just the index, if we want the feature name we have the array
            tree = {feature_indexes[bestFeature]:{}}
            uniqueFeatureValues = list(set(data[:,bestFeature]))
            
            # Find the possible feature values
            for featureValue in uniqueFeatureValues:
                newData = np.empty(shape=(0,data.shape[1]-1),dtype=data.dtype)
                newClasses = np.array([],dtype=classes.dtype)
                newIndexes = np.array([],dtype=int)
                index = 0
                # Find the datapoints with each feature value
                for datapoint in data:
                    if datapoint[bestFeature]==featureValue:
                        if bestFeature==0:
                            datapoint = datapoint[1:]
                            newIndexes = feature_indexes[1:]
                        elif bestFeature==nFeatures:
                            datapoint = datapoint[:-1]
                            newIndexes = feature_indexes[:-1]
                        else:
                            datapoint = np.concatenate((datapoint[:bestFeature], 
                                                        datapoint[bestFeature+1:]), 
                                                        axis=0) 
                            newIndexes = np.concatenate((feature_indexes[:bestFeature],
                                                         feature_indexes[bestFeature+1:]),
                                                         axis=0)
                        newData = np.vstack([newData, datapoint])
                        newClasses = np.append(newClasses,classes[index]) 
                    index += 1
                    
                # Now recurse to the next level
                subtree = self.__make_tree(newData,newClasses,newIndexes) 
                # And on returning, add the subtree on to the tree 
                tree[feature_indexes[bestFeature]][featureValue] = subtree
            return tree

#########################Experimentation Option################################

##############################Voting###########################################
headers = ["party", "handicapped", "water project cost sharing", "adoption of the budget resolution", 
           "physician fee freeze", "el salvador aid","religious groups in schools",
           "anti satellite test ban", "aid to nicaraguan contras", "mx missile",
           "immigration","synfuels corporation cutback","education spending",
           "superfund right to sue","crime","duty free exports","export administration act south africa"]

df = pd.read_csv("./voting.csv",header=None, names=headers, na_values="?" )

#how many rows are nulls?
df.isnull().any(axis=1).sum()
df = df.dropna()

voting_targets_names = df.iloc[:,0]

voting_targets = voting_targets_names.replace({"democrat" : 0,"republican" : 1})
voting_targets = voting_targets.values
df = df.drop(columns=['party'])

df.replace({'n': 0,'y' : 1}, regex=True, inplace=True)

# get data as numpy array
voting_data = df.values
voting_train_data, voting_test_data, voting_train_targets, voting_test_targets = train_test_split(voting_data, voting_targets, test_size=0.3, random_state=42)

# Initialize our classifier
dt = MyDecisionTreeClassifier()

# Train our classifier
dt.fit(voting_train_data, voting_train_targets)

# Make predictions
voting_predicted_targets = dt.predict(voting_test_data)

# Evaluate accuracy
print("Voting My accuracy: %.2f" % accuracy_score(voting_test_targets, voting_predicted_targets))

dt = tree.DecisionTreeClassifier()

# Train our classifier
dt.fit(voting_train_data, voting_train_targets)

# Make predictions
voting_predicted_targets = dt.predict(voting_test_data)

# Evaluate accuracy
print("Voting Built int accuracy: %.2f" % accuracy_score(voting_test_targets, voting_predicted_targets))

#################################Lenses#######################################
            
df = pd.read_csv("./lenses.csv")
targets = df.iloc[:,-1].values
data = df.iloc[:,:3].values

df = pd.read_csv("./lenses.csv", header=None)
lenses_targets = df.iloc[:,-1].values
lenses_data = df.iloc[:,:4].values

lenses_train_data, lenses_test_data, lenses_train_targets, lenses_test_targets = train_test_split(lenses_data, lenses_targets, test_size=0.1, random_state=42)

# Initialize our classifier
dt = MyDecisionTreeClassifier()

# Train our classifier
dt.fit(lenses_train_data, lenses_train_targets)

# Make predictions
lenses_predicted_targets = dt.predict(lenses_test_data)

# Evaluate accuracy
print("Lenses My accuracy: %.2f" % accuracy_score(lenses_test_targets, lenses_predicted_targets))

dt = tree.DecisionTreeClassifier()

# Train our classifier
dt.fit(lenses_train_data, lenses_train_targets)

# Make predictions
lenses_predicted_targets = dt.predict(lenses_test_data)

# Evaluate accuracy
print("Lenses Built int accuracy: %.2f" % accuracy_score(lenses_test_targets, lenses_predicted_targets))

#################################Iris#######################################

#using built in Classifier since the data is continious
iris = datasets.load_iris()

iris_train_data, iris_test_data, iris_train_targets, iris_test_targets  = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
 
# Initialize our classifier
dt = tree.DecisionTreeClassifier()

# Train our classifier
dt.fit(iris_train_data, iris_train_targets)

# Make predictions
predicted_targets = dt.predict(iris_test_data)

# Evaluate accuracy
print("Iris accuracy:  %.2f" % accuracy_score(predicted_targets, iris_test_targets))