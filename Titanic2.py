from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# modules to handle data
import pandas as pd
import numpy as np

train = pd.read_csv('d:/Prabhkirat/Python/Titanic/train.csv')
test = pd.read_csv('d:/Prabhkirat/Python/Titanic/test.csv')

passengerId = test.PassengerId

# merge train and test
X = train.append(test, ignore_index=True)
# create indexes to separate data later on
train_idx = len(train)
test_idx = len(X) - len(test)

#X = train[['Sex','Age','Pclass','Embarked','Ticket','SibSp','Parch', 'Cabin', 'Fare', 'Name']]
#y = train[['Survived']]
#age_groups = {0: (0,15) , 1: (15,25), 2: (25,40), 3: (40,65), 4: (65,81)}
#print(X.shape)
#print(y.shape)
X['Title'] = X.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
#print(X['Title'])
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
X.Title = X.Title.map(normalized_titles)
print(X.Title.value_counts())

# group by Sex, Pclass, and Title 
grouped = X.groupby(['Sex','Pclass', 'Title'])  
# view the median Age by the grouped features 
grouped.Age.median()

X.Age = grouped.Age.apply(lambda x: x.fillna(x.median()))

# fill Cabin NaN with U for unknown
X.Cabin = X.Cabin.fillna('U')
# find most frequent Embarked value and store in variable
most_embarked = X.Embarked.value_counts().index[0]

# fill NaN with most_embarked value
X.Embarked = X.Embarked.fillna(most_embarked)
# fill NaN with median fare
X.Fare = X.Fare.fillna(X.Fare.median())

# view changes
X.info()

# size of families (including the passenger)
X['FamilySize'] = X.Parch + X.SibSp + 1

X.Cabin = X.Cabin.map(lambda x: x[0])

# Convert the male and female groups to integer form
X.Sex = X.Sex.map({"male": 0, "female":1})
# create dummy variables for categorical features
pclass_dummies = pd.get_dummies(X.Pclass, prefix="Pclass")
title_dummies = pd.get_dummies(X.Title, prefix="Title")
cabin_dummies = pd.get_dummies(X.Cabin, prefix="Cabin")
embarked_dummies = pd.get_dummies(X.Embarked, prefix="Embarked")
# concatenate dummy columns with main dataset
X_dummies = pd.concat([X, pclass_dummies, title_dummies, cabin_dummies, embarked_dummies], axis=1)

# drop categorical fields
X_dummies.drop(['Pclass', 'Title', 'Cabin', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

X_dummies.head()

# create train and test data
train = X_dummies[ :train_idx]
test = X_dummies[test_idx: ]

# convert Survived back to int
train.Survived = train.Survived.astype(int)
# create X and y for data and target values 
X = train.drop('Survived', axis=1).values 
y = train.Survived.values
# create array for test set
X_test = test.drop('Survived', axis=1).values

# create param grid object 
forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)

# instantiate Random Forest model
forrest = RandomForestClassifier()

# build and fit model 
forest_cv = GridSearchCV(estimator=forrest,param_grid=forrest_params, cv=5) 
forest_cv.fit(X, y)

print("Best score: {}".format(forest_cv.best_score_))
print("Optimal params: {}".format(forest_cv.best_estimator_))

forrest_pred = forest_cv.predict(X_test)

kaggle = pd.DataFrame({'PassengerId': passengerId, 'Survived': forrest_pred})
# save to csv
kaggle.to_csv('d:/Prabhkirat/Python/Titanic/titanic_pred.csv', index=False)



