from sklearn import tree

import pandas as pd
mydata= pd.read_csv('testing.csv')

training_labels = mydata.iloc[:,0]
mydataframe = mydata.iloc[:, 1:28]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label = le.fit_transform(training_labels)

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 3)
clf = clf.fit(mydataframe, label)

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

# for i in range(len(prediction))  :
#     if prediction[i] != target[i]:
#         print(clf.decision_path(myTest).todense()[i])
#         print('class:'+ target[i])
#         print('predict:' + prediction[i])



print(mydataframe.target_names[prediction])
