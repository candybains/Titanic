import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as train_test_split
from sklearn.linear_model import LogisticRegression
from math import modf

train = pd.read_csv('d:/Prabhkirat/Python/Titanic/train.csv')
test = pd.read_csv('d:/Prabhkirat/Python/Titanic/test.csv')
print(train.shape)
print(test.shape)
print(train.dtypes)

print(train.duplicated().value_counts())
print(train['PassengerId'].is_unique)

survival = {0: 'Not Survived', 1 : 'Survived'}
print(train['Survived'].value_counts())
#print(train['Not Survived'].value_counts())

prop_survived = train['Survived'].value_counts()/len(train)
print(prop_survived)
prop_survived = prop_survived.rename(survival)
#print(prop_survived.index.tolist())
#print(prop_survived.tolist())
plt.figure()
plt.bar(prop_survived.index.tolist(), prop_survived.tolist())
plt.show()
gender_data = train['Sex']
print(gender_data[:10])
print(gender_data.isna().value_counts())

prop_gender = gender_data.value_counts()/len(gender_data)
print(prop_gender)

male_female_survived = train.groupby(by=['Sex','Survived']).size().reset_index().rename(columns={0: 'Count'})
print(male_female_survived)
import seaborn as sns
sns.barplot(x='Sex', y='Count', hue='Survived', data=male_female_survived).set_title('Survival rate by Gender')
plt.show()

prop_male_female_survived = male_female_survived.copy()

prop_male_female_survived['Count'] = prop_male_female_survived['Count']/prop_male_female_survived['Count'].sum()
prop_male_female_survived.rename(columns = {'Count' : 'Percentage'},inplace= True)
print(prop_male_female_survived)

male_survived = train[train['Sex']=='male']
print(male_survived['Survived'].value_counts())
male_survived_val = male_survived['Survived'].value_counts().rename(survival)
plt.figure()
plt.bar(male_survived_val.index.tolist(),male_survived_val.tolist())
plt.title('Survival rate of Male Passengers')
plt.show()

prop_male_survived = male_survived_val/len(male_survived)
print(prop_male_survived)

female_survived = train[train['Sex'] == 'female']
female_survived_val = female_survived['Survived'].value_counts().rename(survival)
plt.figure()
plt.bar(female_survived_val.index.tolist(), female_survived_val.tolist())
plt.title('Survival rate of Female Passengers')
plt.show()

prop_female_survived = female_survived_val/len(female_survived)
print(prop_female_survived)

#Age Groups
print(train['Age'].isna().value_counts())
print(train['Age'].describe())

df_modified = train[['PassengerId', 'Survived', 'Sex', 'Age']].copy()
print(train['Age'][train['Age'].notna()].mean())

df_modified['Age'] = df_modified['Age'].fillna(modf(train['Age'][train['Age'].notna()].mean())[1])
print(df_modified[-10:])

age_groups = {0: (0,15) , 1: (15,25), 2: (25,40), 3: (40,65), 4: (65,81)}

def which_age_group(x):
	for key, age_group in age_groups.items():
		if x>=age_group[0] and x<age_group[1]:
			return key
			
df_modified['Age_group'] = df_modified['Age'].apply(which_age_group).astype('int64')
print(df_modified[:10])

agegroup = df_modified.groupby('Age_group').size().rename(age_groups)
print(agegroup.index.tolist())
print(agegroup.tolist())
plt.figure()
#plt.bar(agegroup.index.tolist(), agegroup.tolist())
#plt.show()


 
