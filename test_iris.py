from sklearn import datasets
iris = datasets.load_iris()

from sklearn import tree
import pandas as pd
dftrain = pd.read_csv('training.csv')

training_labels = dftrain.iloc[:,0]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
label = le.fit_transform(training_labels)


from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np

from sklearn import datasets
iris = datasets.load_iris()
irisTest = np.array([[4.6,3.5,1.1,0.25],[5.7,2.5,2.8,1.2],[7.3,2.8,6.6,2.2]])

clf = RandomForestClassifier(n_estimators = 3)
clf = clf.fit(iris.data, iris.target)
prediction = clf.predict(irisTest)
print(iris.target_names[prediction])

print(clf.estimators_)

# nb = GaussianNB()
# nb = nb.fit(iris.data, iris.target)
# prediction = nb.predict(irisTest)
# print(iris.target_names[prediction])
