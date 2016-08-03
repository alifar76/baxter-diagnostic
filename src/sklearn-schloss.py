from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import recall_score

import numpy as np
import pandas as pd
import math
from collections import Counter



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


clf = RandomForestClassifier(
	n_estimators=7000,random_state=0,criterion='entropy',min_samples_split=20).fit(X, Y.values.ravel())
#print ("Accuracy of Random Forest Classifier: "+str(clf.score(P,Q)))

clf2 = SVC(kernel='rbf',C=1,
	gamma=0.001,random_state=0).fit(X, Y.values.ravel())
#print ("Accuracy of SVM: "+str(clf2.score(P,Q)))


clf3 = GradientBoostingClassifier(n_estimators=1000, learning_rate=1,
max_depth=10, random_state=0, min_samples_split=5).fit(X, Y.values.ravel())
#print ("Accuracy of Gradient Boosting Classifier: "+str(clf3.score(P,Q)))

clf4 = GaussianNB().fit(X, Y.values.ravel())
#print ("Accuracy of Gaussian Naive Bayes Classifier: "+str(clf4.score(P,Q)))


# algorithm, learning_rate_init, alpha, hidden_layer_sizes 
# and activation have impact
clf6 = MLPClassifier(algorithm='adam', alpha=0.01, max_iter=500,
	learning_rate='constant', hidden_layer_sizes=(400,), 
	random_state=0, learning_rate_init=1e-2,
	activation='logistic').fit(X, Y.values.ravel())
#print ("Accuracy of Multi-layer Perceptron Classifier: "+str(clf6.score(P,Q)))



print ("Sensitivity of RF: "+str(recall_score(clf.predict(P),Q)))
print ("Sensitivity of SVM: "+str(recall_score(clf2.predict(P),Q)))
print ("Sensitivity of Gradient Boosting Trees: "+str(recall_score(clf3.predict(P),Q)))
print ("Sensitivity of Gaussian NB: "+str(recall_score(clf4.predict(P),Q)))
print ("Sensitivity of MLP: "+str(recall_score(clf6.predict(P),Q)))


#Yet current screening tests, the fecal immunochemical test (FIT) and the multitarget DNA test, have a sensitivity of 7.6% and 17.2%

#Sensitivity of RF: 0.367346938776
#Sensitivity of SVM: 0.316326530612
#Sensitivity of Gradient Boosting Trees: 0.387755102041
#Sensitivity of Gaussian NB: 0.408163265306
#Sensitivity of MLP: 0.489795918367



#Accuracy of Random Forest Classifier: 0.367346938776
#Accuracy of SVM: 0.316326530612
#Accuracy of Gradient Boosting Classifier: 0.387755102041
#Accuracy of Gaussian Naive Bayes Classifier: 0.408163265306
#Accuracy of Multi-layer Perceptron Classifier: 0.489795918367





## Combine dataframes
#dataf = pd.concat([filtered_data, dx], axis=1)
#print (list(a.columns.values))



