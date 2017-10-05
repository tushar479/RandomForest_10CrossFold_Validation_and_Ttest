from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from scipy import stats

# create data frame containing your data, each column can be accessed  by df[column   name]
df = pd.read_csv('/Users/tushar/Desktop/DAL/Machinelearning_for_bigdata/Qusetion_1/animals.csv', sep=",", index_col=False)
df = df.values
scoring = 'accuracy'
#lets divide the dataset into train and test. test dataset contains the final column with class values
train = np.array(df[:,0:25])
test = np.array(df[:,25:26])

#The parameter test_size is given value 0.3;
# it means test sets will be 30 of whole dataset and training datasets size will be 70 of the entire dataset.
X_train, X_test, y_train, y_test = train_test_split( train, test, test_size = 0.3, random_state = 0)

#Random forest classifier
#Random forest classifier with 10 trees
clf = RandomForestClassifier(n_estimators=10) #10 trees
# 10-Fold Cross validation
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring =scoring)
Rf10_Mean_Acc = scores.mean()
Rf10_Std_Dev = scores.std()
print("Random forest classifier with 10 trees Accuracy: %0.2f (+/- %0.2f)" % (Rf10_Mean_Acc, Rf10_Std_Dev))

#Random forest classifier with 20 trees
clf = RandomForestClassifier(n_estimators=20) #20 trees
# 10-Fold Cross validation
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=20, scoring =scoring)
Rf20_Mean_Acc = scores.mean()
Rf20_Std_Dev = scores.std()
print("Random forest classifier with 20 trees Accuracy: %0.2f (+/- %0.2f)" % (Rf20_Mean_Acc, Rf20_Std_Dev))

#Random forest classifier with 50 trees
clf = RandomForestClassifier(n_estimators=50) #50 trees
# 10-Fold Cross validation
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=50, scoring =scoring)
Rf50_Mean_Acc = scores.mean()
Rf50_Std_Dev = scores.std()
print("Random forest classifier with 50 trees Accuracy: %0.2f (+/- %0.2f)" % (Rf50_Mean_Acc, Rf50_Std_Dev))

#Random forest classifier with 100 trees
clf = RandomForestClassifier(n_estimators=100) #100 trees
# 10-Fold Cross validation
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=100, scoring =scoring)
Rf100_Mean_Acc = scores.mean()
Rf100_Std_Dev = scores.std()
print("Random forest classifier with 100 trees Accuracy: %0.2f (+/- %0.2f)" % (Rf100_Mean_Acc,Rf100_Std_Dev))

kfold = model_selection.KFold(n_splits=10, random_state=None)

#Decision Tree Classifier
clf = DecisionTreeClassifier()
# 10-Fold Cross validation
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=kfold, scoring =scoring)
Dt_Mean_Acc = scores.mean()
Dt_Std_Dev = scores.std()
print("Decision Tree Classifier Accuracy: %0.2f (+/- %0.2f)" % (Dt_Mean_Acc, Dt_Std_Dev))

#Naive Bayes classifier
clf = GaussianNB()
# 10-Fold Cross validation
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=kfold, scoring =scoring)
NB_Mean_Acc = scores.mean()
NB_Std_Dev = scores.std()
print("Naive Bayes classifier Accuracy: %0.2f (+/- %0.2f)" % (NB_Mean_Acc, NB_Std_Dev))

#Logistics Regression classifier
clf = LogisticRegression()
# 10-Fold Cross validation
scores = cross_val_score(clf, X_train, y_train.ravel(), cv=kfold, scoring =scoring)
LR_Mean_Acc = scores.mean()
LR_Std_Dev = scores.std()
print("Logistics Regression classifier Accuracy: %0.2f (+/- %0.2f)" % (LR_Mean_Acc, LR_Std_Dev))




# Given alpha value in question
alpha=0.05


# Calculating statistical test for Mean and standard deviation of accuracy scores in Decision Tree and Random Forest
print("Ttest Random forest vs Decision tree")
t, p = stats.ttest_ind_from_stats(Rf100_Mean_Acc, Rf100_Std_Dev, 100,
                                Dt_Mean_Acc, Dt_Std_Dev, 10,
                                equal_var=False)

print("ttest_ind_from_stats: t = %g  p = %g" % (t, p))

# Determinig the Statistical significance using ttest and alpha
if (p<alpha):
    print("P is less than alpha(0.05), hence there is a significant difference between the Random Forest and Decision Tree classification")
else:
    print("There is no significant difference between the Random Forest and Decision Tree classification")

print("Ttest Random forest vs Naivebayes")
# Calculating statistical test for Mean and standard deviation of accuracy scores in NaiveBayes and Random Forest
t, p = stats.ttest_ind_from_stats(Rf100_Mean_Acc, Rf100_Std_Dev, 100,
                                  NB_Mean_Acc, NB_Std_Dev, 10,
                                  equal_var=False)

print("ttest_ind_from_stats: t = %g  p = %g" % (t, p))

# Determinig the Statistical significance using ttest and alpha
if (p<alpha):
    print("P is less than alpha(0.05), hence there is a significant difference between the Random Forest and NaiveBayes classification")
else:
    print("There is no significant difference between the Random Forest and NaiveBayes classification")

print("Ttest Random forest vs Logistics Regression")
# Calculating statistical test for Mean and standard deviation of accuracy scores in Logistics Regression and Random Forest
t, p = stats.ttest_ind_from_stats(Rf100_Mean_Acc, Rf100_Std_Dev, 100,
                                  LR_Mean_Acc, LR_Std_Dev, 10,
                                  equal_var=False)

print("ttest_ind_from_stats: t = %g  p = %g" % (t, p))

# Determinig the Statistical significance using ttest and alpha
if (p<alpha):
    print("P is less than alpha(0.05), hence there is a significant difference between the Random Forest and Logistics Regression classification")
else:
    print("There is no significant difference between the Random Forest and Logistics Regression classification")


