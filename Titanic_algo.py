import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from math import modf
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv('d:/Prabhkirat/Python/Titanic/train.csv')
test = pd.read_csv('d:/Prabhkirat/Python/Titanic/test.csv')
print(train.shape)
print(test.shape)
print(train.columns)


normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#X = train[['Sex','Age','Pclass','Embarked','Ticket','SibSp','Parch', 'Cabin', 'Fare', 'Name']]
X = train.append(test)
X.reset_index(inplace=True)
X.drop(['index', 'PassengerId','Survived'], inplace=True, axis=1)
y = train[['Survived']]
age_groups = {0: (0,15) , 1: (15,25), 2: (25,40), 3: (40,65), 4: (65,81)}
#print(X.shape)
#print(y.shape)
X['Fare'] = X['Fare'].fillna(modf(X['Fare'][X['Fare'].notna()].mean())[1])
#print(X[-10:])
X.Sex[X.Sex =='female'] = 1
X.Sex[X.Sex == 'male'] = 0
#print(X)
#print(X[X.columns].mean())
#X = X.fillna(X[X.columns].mean())
#print(X.info())
#plt.figure()
#pd.plotting.scatter_matrix(X,figsize=(15,15), grid=False);
#plt.show()
#print(X.Embarked.unique())
X.Embarked = X.Embarked.fillna('S')
#X.Cabin = X.Cabin.fillna(X.Cabin.dropna().mode()[0])
#X['Cabin'] = X['Cabin'].str.get(0)
#print(X.Cabin.unique())

for index, row in X.iterrows():
	if pd.isna(row['Cabin'])== False:
		for index1, row1 in X.iterrows():
			if  row['Ticket'] == row1['Ticket'] and row['Embarked'] == row1['Embarked']:
				if pd.isna(row1['Cabin'])== True:
					X.Cabin[index1] = row['Cabin']
	if pd.isna(row['Cabin'])== True:
		if row['SibSp'] == 0 and row['Parch'] == 0:
			X.Cabin[index] = 'I'
		if row['SibSp'] == 0 and row['Parch'] > 0:
			X.Cabin[index] = 'J'
		if row['SibSp'] > 0 and row['Parch'] == 0:
			X.Cabin[index] = 'K'
		if row['SibSp'] > 0 and row['Parch'] > 0:
			X.Cabin[index] = 'L'
			
X['Cabin'] = X['Cabin'].str.get(0)			
X['Cabin'] = X.Cabin.fillna('U')

X['Ticket'] = X['Ticket'].str.replace('.','')
X['Ticket'] = X['Ticket'].str.replace('/','')
X['Ticket'] = X.Ticket.apply(lambda t: t.split(' ')[0].strip())
X['Ticket'][X['Ticket'].str.isdigit()] = 'XXX'
#print(X['Ticket'])
X['Ticket_len'] = X.Ticket.apply(len)
#print(X.Ticket_len.value_counts())
X['Family'] = X.SibSp + X.Parch
X['Family'] = X['Family'].replace([0],'Alone')
X['Family'] = X['Family'].replace([1,2,3],'Little Family')
X['Family'] = X['Family'].replace([4,5,6,7,8,9,10],'Big Family')

X['Title'] = X.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
X.Title = X.Title.map(normalized_titles)
	
grouped = X.groupby(['Sex','Pclass', 'Title'])  
grouped.Age.median()
X.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

pclass_dummies = pd.get_dummies(X.Pclass, prefix="Pclass")
title_dummies = pd.get_dummies(X.Title, prefix="Title")
cabin_dummies = pd.get_dummies(X.Cabin, prefix="Cabin")
embarked_dummies = pd.get_dummies(X.Embarked, prefix="Embarked")
ticket_dummies = pd.get_dummies(X.Ticket, prefix="Ticket")
family_dummies = pd.get_dummies(X.Family, prefix="Family")
# concatenate dummy columns with main dataset
X = pd.concat([X, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies, ticket_dummies, family_dummies], axis=1)

X.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket', 'Family'], axis=1, inplace=True)
#print(X.Family.value_counts())
#X['Ticket_letter'] = X.Ticket.str[0]
#print(X.Ticket_letter.value_counts())
test_X = X.iloc[891:]
X = X.iloc[:891]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)
#print(train.describe(include=['O']))

#Logistic Regression
#logis = LogisticRegression(penalty='l1', C = 5)
#logis.fit(X_train,y_train)
#y_pred = logis.predict(X_test)
#print(y_pred)
#print('Accuracy of logistic regression classifier on training set: {:.2f}'.format(logis.score(X_train, y_train)))
#print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logis.score(X_test, y_test)))
#print('Confusion Matrix: ',confusion_matrix(y_test,y_pred)) 
#testy_pred = logis.predict(test_X)

#submission = test.copy()
#submission['Survived'] = testy_pred
#print(submission[['PassengerId', 'Survived']].head(15))

#plt.figure()
#sns.regplot(X_test.Age, y_test.Survived)
#sns.regplot(X_test.Age, y_pred)
#plt.show()

# Support Vector Machine
#svm = SVC(kernel = 'rbf', gamma = 'scale', C = 15)
#svm.fit(X_train,y_train)
#y_svm_pred = svm.predict(X_test)
#print(y_svm_pred)
#print('Accuracy of SVM on training set: {:.2f}'.format(svm.score(X_train, y_train)))
#print('Accuracy of SVM on test set: {:.2f}'.format(svm.score(X_test, y_test)))
#print('Confusion Matrix: ',confusion_matrix(y_test,y_pred)) 
#plt.figure()
#sns.regplot(X_test.Age, y_test.Survived)
#sns.regplot(X_test.Age, y_pred)
#plt.show()

forrest_params = dict(     
	max_depth = [4, 6, 8],
	n_estimators = [50, 10],
	max_features = ['sqrt', 'auto', 'log2'],
	min_samples_split = [2, 3, 10],
	min_samples_leaf = [1, 3, 10],
	bootstrap = [True, False]
)

rf = RandomForestClassifier( n_estimators = 170, random_state = 20, max_depth = 40, min_samples_split = 10)
#forest = RandomForestClassifier()
#cross_validation = StratifiedKFold(n_splits=5)
#rf = GridSearchCV(forest,scoring='accuracy', param_grid=forrest_params, cv=cross_validation,verbose=1)
rf.fit(X_train, y_train)
#print(rf.decision_path(X_train))
y_nb_predict = rf.predict(X_test)
#print("Best score: {}".format(rf.best_score_))
print('Accuracy of RF on training set: {:.2f}'.format(rf.score(X_train, y_train)))
print('Accuracy of RF on test set: {:.2f}'.format(rf.score(X_test, y_test)))
#nn = MLPClassifier(hidden_layer_sizes=(10,25,50), activation='tanh', alpha = 1, random_state = 1, solver='lbfgs')
#nn.fit(X_train, y_train)
#y_nn_predict = nn.predict(X_test)
#print('Accuracy of NN on training set: {:.2f}'.format(nn.score(X_train, y_train)))
#print('Accuracy of NN on test set: {:.2f}'.format(nn.score(X_test, y_test)))
testy_pred = rf.predict(test_X)
submission_svm = test.copy()
submission_svm['Survived'] = testy_pred
print(submission_svm[['PassengerId', 'Survived']].head(15))
submission_svm.to_csv('d:/Prabhkirat/Python/Titanic/submission.csv', columns=['PassengerId', 'Survived'], index=False)

#features = pd.DataFrame()
#features['feature'] = X_train.columns
#features['importance'] = rf.feature_importances_
#features.sort_values(by=['importance'], ascending=True, inplace=True)
#features.set_index('feature', inplace=True)
#features.plot(kind='barh')
#plt.show()