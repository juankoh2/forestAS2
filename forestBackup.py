from sklearn import tree

import pandas as pd
mydata= pd.read_csv('testing.csv')

training_labels = mydata.iloc[:,0]
mydataframe = mydata.iloc[:, 1:28]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label = le.fit_transform(training_labels)

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

n = 3;

clf = RandomForestClassifier(n_estimators = n, max_depth=2)
# clf = GaussianNB()
clf = clf.fit(mydataframe, label)

import graphviz

for x in range(0,n):
	dot_data = tree.export_graphviz(clf.estimators_[x], out_file=None,
	feature_names = mydataframe.columns,
	class_names=["Sugi","Hinoki","mixed deciduous","non-forest land"],
	filled=True, rounded=True,
	special_characters=True)

	graph = graphviz.Source(dot_data)
	graph.view('image' + x)




import numpy as np

myTest = pd.read_csv('training.csv')
myTestData = myTest.iloc[:,1:28]
prediction = clf.predict(myTestData)
target = myTest.iloc[:,0]
target = le.fit_transform(target)

print(clf.estimators_)

from sklearn.metrics import accuracy_score, confusion_matrix
print (accuracy_score(target,prediction))
print (confusion_matrix(target,prediction))

# print(mydataframe.target_names[prediction])

