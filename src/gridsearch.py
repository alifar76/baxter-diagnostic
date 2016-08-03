import numpy as np
import pandas as pd
import math
from datetime import datetime

from sklearn.grid_search import GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

def grid_search(model,tuned_parameters,X,Y):
    startTime = datetime.now()
    model_gs = GridSearchCV(model, 
	tuned_parameters, cv=10)
    model_gs.fit(X, Y.values.ravel())
    print("Best parameters set found on development set:")
    print(model_gs.best_params_)
    print "\n"+"Task Completed! Completion time: "+ str(datetime.now()-startTime)
    return


# Load data
# 335 OTUs in total and 490 samples
otuinfile = 'glne007.final.an.unique_list.0.03.subsample.0.03.filter.shared'
mapfile = 'metadata.tsv'
disease_col = 'dx'
data = pd.read_table(otuinfile,sep='\t',index_col=1)
filtered_data = data.dropna(axis='columns', how='all')
filtered_data = filtered_data.drop(['label','numOtus'],axis=1)
metadata = pd.read_table(mapfile,sep='\t',index_col=0)
dx = metadata[disease_col]
X, P, Y, Q = train_test_split(
filtered_data, dx, test_size=0.2, random_state=42)

# Estimate parameters

rf_pg = [{'n_estimators':[1000,2000,3000,4000,5000],'random_state':[0]}]

svm_pg = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'random_state':[0]},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 
  'kernel': ['rbf'], 'random_state':[0]},
 ]

gbc_pg = [{'n_estimators':[1000,2000,3000,4000,5000],
'learning_rate':[100,10,1,1e-2,1e-3,1e-4,1e-5],
'max_depth':[10,100,1000],
'min_samples_split':[5,10,15,20],
'random_state':[0]}]

mlp_pg = [{'algorithm': ['adam','sgd','l-bfgs'], 
  'alpha': [1,1e-1,1e-2,1e-3, 1e-4,1e-5],
  'hidden_layer_sizes': [(100,), (200,), (300,), (400,),(500,)],
  'max_iter':[500], 'random_state':[0],
  'learning_rate' : ['constant', 'invscaling', 'adaptive']}]

# Grid search
# No grid search for Gaussian Naive Bayes
grid_search(RandomForestClassifier(),rf_pg,X,Y)
#{'n_estimators': 2000, 'random_state': 0}
#Task Completed! Completion time: 0:07:20.888383
print ("RF complete")
grid_search(SVC(),svm_pg,X,Y)
#{'kernel': 'rbf', 'C': 1, 'random_state': 0, 'gamma': 0.001}
#Task Completed! Completion time: 0:00:17.674036
print ("SVM complete")
grid_search(GradientBoostingClassifier(),gbc_pg,X,Y)

print ("GBC complete")
grid_search(MLPClassifier(),mlp_pg,X,Y)

print ("MLP complete")
