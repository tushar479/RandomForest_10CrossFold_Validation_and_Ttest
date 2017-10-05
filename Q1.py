from sklearn.naive_bayes import GaussianNB
from sklearn.tree.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# create data frame containing your data, each column can be accessed  by df[column   name]
df = pd.read_csv('/Users/tushar/Desktop/DAL/MachineLearning_for_bigdata/Qusetion_1/animals.csv', sep=',', index_col=False)
df = df.values

#let's divide the dataset into train and test. test dataset contains the final column with class values
train = np.array(df[:,0:25])
test = np.array(df[:,25:26])

#The parameter test_size is given value 0.3;
# it means test sets will be 30 of whole dataset and training datasets size will be 70 of the entire dataset.
X_train, X_test, y_train, y_test = train_test_split( train, test, test_size = 0.3, random_state = 0)

#Naive bayes classifier
clf = GaussianNB()
trained_model_NB = clf.fit(X_train, y_train.ravel())
NB_predict = clf.predict(X_test)

print "Naive bayes classifier:"
# Train and Test Accuracy
print "Train Accuracy :: ", accuracy_score(y_train, trained_model_NB.predict(X_train))
print ""
# Test confusion matrix
print "Train Confusion Matrix"
print confusion_matrix(y_train,trained_model_NB.predict(X_train))
print ""

print "Test Accuracy  :: ", accuracy_score(y_test, NB_predict)
print ""
# Test confusion matrix
print "Test Confusion Matrix"
print confusion_matrix(y_test,NB_predict)
print ""

#Decision tree Classifier
clf = DecisionTreeClassifier()
trained_model_DC = clf.fit(X_train,y_train.ravel())
y_predict = clf.predict(X_test)

print "Decision tree Classifier:"
# Train and Test Accuracy
print "Train Accuracy :: ", accuracy_score(y_train, trained_model_DC.predict(X_train))
print ""
# Test confusion matrix
print "Train Confusion Matrix"
print confusion_matrix(y_train,trained_model_DC.predict(X_train))
print ""

print "Test Accuracy  :: ", accuracy_score(y_test, y_predict)
print ""
# Test confusion matrix
print "Test Confusion Matrix"
print confusion_matrix(y_test,y_predict)
print ""

#Random forest classifier
clf = RandomForestClassifier()
trained_model = clf.fit(X_train, y_train.ravel())
predictions = trained_model.predict(X_test)

print "Random forest classifier:"
# Train Accuracy
print "Train Accuracy :: ", accuracy_score(y_train, trained_model.predict(X_train))
print ""
# Test confusion matrix
print "Train Confusion Matrix"
print confusion_matrix(y_train,trained_model.predict(X_train))
print ""

# Test Accuracy
print "Test Accuracy  :: ", accuracy_score(y_test, predictions)
print ""
# Test confusion matrix
print "Test Confusion Matrix"
print confusion_matrix(y_test,predictions)
print ""

#Logistics regression
logistic_regression_model = LogisticRegression()
LR_trained = logistic_regression_model.fit(X_train,y_train.ravel())
LR_predict = logistic_regression_model.predict(X_test)

print "Logistics regression Classifier:"
# Train and Test Accuracy
print "Train Accuracy :: ", accuracy_score(y_train, LR_trained.predict(X_train))
print ""
# Test confusion matrix
print "Train Confusion Matrix"
print confusion_matrix(y_train,LR_trained.predict(X_train))
print ""

print "Test Accuracy  :: ", accuracy_score(y_test, LR_predict)
print ""
# Test confusion matrix
print "Test Confusion Matrix"
print confusion_matrix(y_test,LR_predict)
print ""



