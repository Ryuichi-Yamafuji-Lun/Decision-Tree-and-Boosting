import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Part 2: Implementation of AdaBoost with decision trees as weak learners

class AdaBoost:
  def __init__(self, n_estimators=60, max_depth=10):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.betas = []
    self.models = []
    
  def fit(self, X, y):
    ###########################TODO#############################################
    # In this part, please implement the adaboost fitting process based on the 
    # lecture and update self.betas and self.models, using decision trees with 
    # the given max_depth as weak learners

    # Inputs: X, y are the training examples and corresponding (binary) labels
    
    # Hint 1: remember to convert labels from {0,1} to {-1,1}
    # Hint 2: DecisionTreeClassifier supports fitting with a weighted training set
        
    return self
    
  def predict(self, X):
    ###########################TODO#############################################
    # In this part, make prediction on X using the learned ensemble
    # Note that the prediction needs to be binary, that is, 0 or 1.
    
    return preds
    
  def score(self, X, y):
    accuracy = accuracy_score(y, self.predict(X))
    return accuracy

