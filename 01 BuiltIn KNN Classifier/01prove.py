# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

iris = datasets.load_iris()
print(iris.data)
#TODO how to randomize
train, test, train_labels, test_labels = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)

# Make predictions
targets_predicted = gnb.predict(test)

# Evaluate accuracy
print("%.2f" % accuracy_score(test_labels, targets_predicted))

#5. IMPLEMENT YOUR OWN NEW "ALGORITHM"
class HardCodedClassifier:
    def fit(self, data_train, targets_train):
        print("Do nothing for now")
        
    def predict(self, data_test):    
        return [0 for data in data_test]
    
# Initialize our classifier
hcc = HardCodedClassifier()

# Train our classifier
model = hcc.fit(train, train_labels)

# Make predictions
targets_predicted = hcc.predict(test)

# Evaluate accuracy
print("%.2f" % accuracy_score(test_labels, targets_predicted))

#STRECH CHALLENGE

#load data
iris_data_frame = pd.read_csv("./iris_data.csv")

print(iris_data_frame.columns)

#check for nulls
iris_data_frame.isnull().values.any()

iris_names = iris_data_frame.iloc[:,-1]

species_map = {"setosa" : 0, "versicolor" : 1, "virginica" : 2}
iris_target = iris_target_names.map(species_map)

train, test, train_labels, test_labels = train_test_split(iris_data, iris_target, test_size=0.3, random_state=42)

# Initialize our classifier
gnb = GaussianNB()

# Train our classifier
model = gnb.fit(train, train_labels)

# Make predictions
targets_predicted = gnb.predict(test)

# Evaluate accuracy
print("%.2f" % accuracy_score(test_labels, targets_predicted))
