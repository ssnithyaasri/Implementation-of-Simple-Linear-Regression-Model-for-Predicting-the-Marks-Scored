# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: NITHYAA SRI S S
RegisterNumber: 212222230100
/*

```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

\df=pd.read_csv('/content/student_scores.csv')
df.head(10)

plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')


x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression


lr=LinearRegression()
lr.fit(X_train,Y_train)

X_train
Y_train

lr.predict(X_test.iloc[0].values.reshape(1,1))4



plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')


m=lr.coef_
m[0]

b=lr.intercept_
b



```
## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![output](ex-1%2001.png)
![output](ex-2%2002.png)
![output](ex-2%2003.png)
![output](ex-2%2004.png)
![output](ex-2%2005.png)
![output](ex-2%2006.png)
![output](ex%20-2%2007.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
