"""
Created on Mon Feb 25 02:08:56 2019

@author: klevindoda
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class MyMLPClassifier:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,hiddenLayers=[5,2],niterations=200,eta=0.2,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        assert len(hiddenLayers) != 0
        #array on integers, where the ith element represents 
        #the number of neurons in the ith hidden layer
        self.hiddenLayers = hiddenLayers
        self.niterations = niterations
        self.eta = eta
        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    	 
    def fit(self,inputs,targets):   
        
        self.nin = np.shape(inputs)[1]   # number of features (columns) of input
        self.nout = np.shape(targets)[1] # number of targets (ex iris has three targets)
        
        
        self.weights = []
        # first layer
        self.weights.append((np.random.rand(self.nin+1,self.hiddenLayers[0])-0.5)*2/np.sqrt(self.nin))
        #the rest of hidden layers
        for i in range(1,len(self.hiddenLayers)):
            self.weights.append((np.random.rand(self.hiddenLayers[i-1]+1,self.hiddenLayers[i])-0.5)*2/np.sqrt(self.hiddenLayers[i-1]))
        #last layer
        self.weights.append((np.random.rand(self.hiddenLayers[-1]+1,self.nout)-0.5)*2/np.sqrt(self.hiddenLayers[-1]))
        
        ndata = np.shape(inputs)[0]
        # Add the column that match the bias node
        inputs = np.concatenate((inputs,-np.ones((ndata,1))),axis=1)
        updates = []
        
        #for testing purposes
# =============================================================================
#         errors = []
#         accuracies = []
# =============================================================================
        
        for weight in self.weights:
            updates.append(np.zeros((np.shape(weight))))
        
        
        for n in range(self.niterations):
    
            self.outputs = self.forward(inputs)
        
            # measure errors in every iterations
# =============================================================================
#             error = 0.5*np.sum((self.outputs-targets)**2)
#             errors.append(error)
# =============================================================================


            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs-targets)/ndata
            elif self.outtype == 'logistic':
                deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/ndata 
            else:
                print ("error")
            
        
            #last layer here since it is special
            updates[-1] = self.eta*(np.dot(np.transpose(self.activations[-1]),deltao)) + self.momentum*updates[-1]
            self.deltah = self.activations[-1]*self.beta*(1.0-self.activations[-1])*(np.dot(deltao,np.transpose(self.weights[-1])))
            
            for i in range(len(self.activations) - 2,-1,-1): 
                updates[i+1] = self.eta*(np.dot(np.transpose(self.activations[i]),self.deltah[:,:-1])) + self.momentum*updates[i+1]
                self.deltah = self.activations[i]*self.beta*(1.0-self.activations[i])*(np.dot(self.deltah[:,:-1],np.transpose(self.weights[i+1])))

            #-1 because of the bias
            #first layer
            updates[0] = self.eta*(np.dot(np.transpose(inputs),self.deltah[:,:-1])) + self.momentum*updates[0]
            
            for i in range(len(self.weights)):
                self.weights[i] -= updates[i]
                
            # meausure accuracies in every iteration for checking the progress
# =============================================================================
#             targets_predicted = np.zeros_like(self.outputs)
#             targets_predicted[np.arange(len(self.outputs)), self.outputs.argmax(1)] = 1
#             accuracies.append(accuracy_score(targets_predicted,targets))  
# =============================================================================
        
        #Error: plot them 
# =============================================================================
#         plt.plot(range(self.niterations), errors, label="errors")
#         plt.ylabel("Error")
#         plt.xlabel("Iteration")
#         plt.legend()  
# =============================================================================
    
        
        #Accuracies: plot them
# =============================================================================
#         plt.plot(range(self.niterations), accuracies, label="accuracies")
#         plt.ylabel("Accuracy")
#         plt.xlabel("Iteration")
#         plt.legend()
# =============================================================================

    def forward(self,inputs):
        """ Run the network forward """

        #activations for each hidden layer
        self.activations = []
        
        # since we are doing a batch update store the activations of each row   
        #first hidden layer
        activation = np.dot(inputs,self.weights[0]); # stores h for each row, for each node 1st hidden layer
        activation = 1.0/(1.0+np.exp(-self.beta*activation)) # stores a for each row, for each node 1st hidden layer
        activation = np.concatenate((activation,-np.ones((np.shape(inputs)[0],1))),axis=1) # add the bias 
        self.activations.append(activation)
        
        for i in range(1,len(self.hiddenLayers)):
            #ith hidden layer
            activation = np.dot(self.activations[i-1],self.weights[i]); # stores h for each row, for each node ith hidden layer
            activation = 1.0/(1.0+np.exp(-self.beta*activation)) # stores a for each row, for each node ith hidden layer
            activation = np.concatenate((activation,-np.ones((np.shape(inputs)[0],1))),axis=1) # add the bias 
            self.activations.append(activation)

        outputs = np.dot(self.activations[-1],self.weights[-1]);

        # Different types of output neurons
        if self.outtype == 'linear':
            	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print ("error")
    
    def predict(self,inputs):  
        ndata = np.shape(inputs)[0]
        # Add the column that match the bias node
        inputs = np.concatenate((inputs,-np.ones((ndata,1))),axis=1)
        outputs = self.forward(inputs)
        targets = np.zeros_like(outputs)
        targets[np.arange(len(outputs)), outputs.argmax(1)] = 1
        return targets

#################################Iris#########################################
iris = np.loadtxt('iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = np.concatenate((iris.max(axis=0)*np.ones((1,5)),np.abs(iris.min(axis=0)*np.ones((1,5)))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

# Split into training, validation, and test sets
target = np.zeros((np.shape(iris)[0],3));
indices = np.where(iris[:,4]==0) 
target[indices,0] = 1
indices = np.where(iris[:,4]==1)
target[indices,1] = 1
indices = np.where(iris[:,4]==2)
target[indices,2] = 1

# Randomly order the data
order = list(range(np.shape(iris)[0]))
np.random.shuffle(order)
iris = iris[order,:]
target = target[order,:]

train = iris[::2,0:4]
traint = target[::2]
valid = iris[1::4,0:4]
validt = target[1::4]
test = iris[3::4,0:4]
testt = target[3::4]


clf = MyMLPClassifier(eta=0.2,niterations=200,hiddenLayers=[5,2],outtype='logistic')
clf.fit(train,traint)
my_predicted_targets = clf.predict(test)

#measuring against the builts in Classifier
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(3), random_state=1)

clf.fit(train,traint)
builtin_predicted_targets = clf.predict(test)

print("Iris My accuracy: %.2f" % accuracy_score(my_predicted_targets,builtin_predicted_targets))

#################################Lenses#######################################
df = pd.read_csv("./lenses.csv", header=None)
lenses_targets = df.iloc[:,-1].values
lenses_data = df.iloc[:,:4].values

# Split into training, validation, and test sets
target = np.zeros((np.shape(lenses_data)[0],3));
indices = np.where(lenses_targets==1) 
target[indices,0] = 1
indices = np.where(lenses_targets==2)
target[indices,1] = 1
indices = np.where(lenses_targets==3)
target[indices,2] = 1

lenses_train_data, lenses_test_data, lenses_train_targets, lenses_test_targets = train_test_split(lenses_data, target, test_size=0.1, random_state=42)

my_predicted_targets = clf.predict(test)

# Initialize our classifier
clf = MyMLPClassifier(eta=0.2,niterations=200,hiddenLayers=[5,2],outtype='logistic')

# Train our classifier
clf.fit(lenses_train_data,lenses_train_targets)

# Make predictions
lenses_predicted_targets = clf.predict(lenses_test_data)

# Evaluate accuracy
print("Lenses My accuracy: %.2f" % accuracy_score(lenses_test_targets, lenses_predicted_targets))

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

# Split into training, validation, and test sets
target = np.zeros((np.shape(voting_data)[0],2));
indices = np.where(voting_targets==0) 
target[indices,0] = 1
indices = np.where(voting_targets==1)
target[indices,1] = 1

voting_train_data, voting_test_data, voting_train_targets, voting_test_targets = train_test_split(voting_data, target, test_size=0.3, random_state=42)

# Initialize our classifier
clf = MyMLPClassifier(eta=0.2,niterations=200,hiddenLayers=[5,2],outtype='logistic')

# Train our classifier
clf.fit(voting_train_data,voting_train_targets)

# Make predictions
voting_predicted_targets = clf.predict(voting_test_data)

# Evaluate accuracy
print("Voting My accuracy: %.2f" % accuracy_score(voting_predicted_targets, voting_test_targets))
