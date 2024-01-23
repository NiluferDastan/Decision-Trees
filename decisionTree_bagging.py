


import pandas as pd

dataset = pd.read_csv("diabetes.csv")

dataset.head()

col_names=["pregnant","glucose","bp","skin","insulin","bmi","pedigree","age","label"]
dataset.columns = col_names

dataset.head()

feature_cols = ["pregnant","glucose","insulin","bmi","age","glucose","bp","pedigree"]

X = dataset[feature_cols]
y = dataset.label

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)

from sklearn import metrics

print("Accuracy : ", metrics.accuracy_score(y_test,y_pred))

"""
bagging classifier

original dataset : 1,2,3,4,5,6,7,8,9,10
resampling dataset : 2,3,4,5,6,1,8,10,9,1
resamplig again : 4,5,5,7,8,1,2,3,9,10 
"""

from sklearn.ensemble import BaggingClassifier
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=3,random_state=8, shuffle=True)
num_trees = 500

model = BaggingClassifier(base_estimator=dtc, n_estimators=num_trees)

result = model_selection.cross_val_score(model, X, y, cv=kfold)

print("Accuracy : ", result.mean())