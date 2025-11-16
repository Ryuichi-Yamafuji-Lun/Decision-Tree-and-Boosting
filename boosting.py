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

    # gets shape of X
    samples = X.shape[0]

    # initialize sample weights
    w = np.full(samples, 1/samples )

    # convert labels 
    new_y = np.where(y == 0, -1, 1)

    for _ in range(self.n_estimators):

      # obtain a weak classifier
      model = DecisionTreeClassifier(max_depth=self.max_depth)
      model.fit(X, y, sample_weight=w)
      h_t = model.predict(X)
      h_t= np.where(h_t == 0, -1, 1)

      # calculate importance of h
      # weighted error epsilon
      error = (h_t != new_y)
      e = np.sum(w[error])
      # beta
      beta = 0.5 * np.log((1 - e)/ e)

      # update distributions
      w *= np.exp(-beta * new_y * h_t)

      w /= np.sum(w)

      self.models.append(model)
      self.betas.append(beta)
    
    return self
    
  def predict(self, X):
    ###########################TODO#############################################
    # In this part, make prediction on X using the learned ensemble
    # Note that the prediction needs to be binary, that is, 0 or 1.
    final_prediction = np.zeros(X.shape[0])

    for t in range(self.n_estimators):
      model, beta = self.models[t], self.betas[t]
      h_t = model.predict(X)
      h_t = np.where(h_t == 0, -1, 1)
      final_prediction += beta * h_t
    
    preds = np.where(final_prediction > 0, 1, 0)
    return preds
    
  def score(self, X, y):
    accuracy = accuracy_score(y, self.predict(X))
    return accuracy

