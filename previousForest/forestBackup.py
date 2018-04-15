from sklearn import tree

import pandas as pd
mydata= pd.read_csv('training.csv')

training_labels = mydata.iloc[:,0]
mydataframe = mydata.iloc[:, 1:28]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label = le.fit_transform(training_labels)

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

n = 12;

clf = RandomForestClassifier(n_estimators = n)
# clf = GaussianNB()
clf = clf.fit(mydataframe, label)

import graphviz

# for x in range(0,n):
# 	dot_data = tree.export_graphviz(clf.estimators_[x], out_file=None,
# 	feature_names = mydataframe.columns,
# 	class_names=["Sugi","Hinoki","mixed deciduous","non-forest land"],
# 	filled=True, rounded=True,
# 	special_characters=True)

# 	graph = graphviz.Source(dot_data)
# 	graph.view('imagea' + str(x))




import numpy as np

myTest = pd.read_csv('testing.csv')
myTestData = myTest.iloc[:,1:28]
prediction = clf.predict(myTestData)
target = myTest.iloc[:,0]
target = le.fit_transform(target)

print(clf.estimators_)

from sklearn.metrics import accuracy_score, confusion_matrix
print (accuracy_score(target,prediction))
print (confusion_matrix(target,prediction))

# print(mydataframe.target_names[prediction])


print(np.argmax(clf.feature_importances_))

import matplotlib.pyplot as plt

from sklearn.datasets import make_classification



importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(mydataframe.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(mydataframe.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(mydataframe.shape[1]), indices)
plt.xlim([-1, mydataframe.shape[1]])
plt.show()