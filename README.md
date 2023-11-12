# Ex:07 Feature Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## EXPLANATION
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

## ALGORITHM
1.Read the given Data
2.Clean the Data Set using Data Cleaning Process
3.Apply Feature selection techniques to all the features of the data set
4.Save the data to the file

## CODE
```
Name: JENISHA.J
Reg.No: 212222230056
```
```
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
```
#### Data loading
```
data = pd.read_csv('/content/titanic_dataset.csv')
data
data.tail()
data.isnull().sum()
data.describe()
```
#### start with a pairplot, and check missing values
```
sns.heatmap(data.isnull(),cbar=False)
```
#### Data Cleaning and Data Drop Process
```
data['Fare'] = data['Fare'].fillna(data['Fare'].dropna().median())
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
```
#### Change to categoric column to numeric
```
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
```
#### Instead of nan values
```
data['Embarked']=data['Embarked'].fillna('S')
```
#### Change to categoric column to numeric
```
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
```
#### Drop unnecessary columns
```
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
data.head(11)
```
#### Heatmap for train dataset
```
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
```
#### Now, data is clean and read to a analyze
```
sns.heatmap(data.isnull(),cbar=False)
```
#### How many people survived or not... %60 percent died %40 percent survived
```
fig = plt.figure(figsize=(18,6))
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
```
#### Age with survived
```
plt.scatter(data.Survived, data.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()
```
#### Count the pessenger class
```
fig = plt.figure(figsize=(18,6))
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = data.drop("Survived",axis=1)
y = data["Survived"]

mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
```
#### Split the data into training and test sets
```
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
```
#### Create a Random Forest classifier
```
my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5, criterion='entropy')
```
#### Fit the model to the training data
```
my_forest.fit(X_train, y_train)
```
#### Make predictions on the test data
```
target_predict = my_forest.predict(X_test)
```
#### Evaluate the model's performance
```
accuracy = accuracy_score(y_test, target_predict)
mse = mean_squared_error(y_test, target_predict)
r2 = r2_score(y_test, target_predict)

print("Random forest accuracy: ", accuracy)
print("Mean Squared Error (MSE): ", mse)
print("R-squared (R2) Score: ", r2)
```

## OUTPUT
#### Initial data
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/0f18dc3c-b5a9-469f-abf7-49a829a46ea4)

#### Null values
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/df3b4c1a-b5c4-4166-a757-c20af0577af4)

#### Describing the data
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/fe9062ca-fff8-45a8-a369-8474a5d58963)

#### Missing values
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/501d95b7-a483-4278-ac68-ab399d2aa9f8)

#### Data after cleaning
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/9463d8bf-4f7a-486f-990f-76bc0e979cef)

#### Data on Heatmap
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/69ec7bb1-9327-44c0-9963-7f304fb6977a)

#### Report of(people survied & died)
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/743e157d-8d36-46fd-ad4b-a25a40046d8f)

#### Cleaned null values
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/22ecf14d-f4c3-4466-ba83-e1b8d18e4670)

#### Report of survied people's age
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/4cd0f502-cd12-4e47-aef0-14593129b6d5)

#### Report of passengers
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/8063ccf3-e567-4ed7-8d71-6c65dba3c130)

#### Report
![image](https://github.com/Jenishajustin/ODD2023-Datascience-Ex-07/assets/119405070/643f502d-e3f3-4a42-bf91-f7fd7875bcd0)

## RESULT
Thus, Successfully performed the various feature selection techniques on a given dataset.
