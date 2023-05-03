# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown values.


## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by:  Divyashree B S
RegisterNumber:  212221040044

import pandas as pd
data=pd.read_csv('/content/Placement_Data (2).csv')
print("Placement Data:")
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)#removes the specified row or col
print("Salary Data:")
data1.head()

print("Checking the null() function:")
data1.isnull().sum()

print ("Data Duplicate:")
data1.duplicated().sum()

print("Print Data:")
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

print("Data-status value of x:")
x=data1.iloc[:,:-1]
x

print("Data-status value of y:")
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

print ("y_prediction array:")
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear") #A Library for Large
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score =(TP+TN)/
#accuracy_score(y_true,y_pred,normalize=False)
print("Accuracy value:")
accuracy

from sklearn.metrics import confusion_matrix 
confusion=(y_test,y_pred) 
print("Confusion array:")
confusion

from sklearn.metrics import classification_report 
classification_report1=classification_report(y_test,y_pred) 
print("Classification report:")
print(classification_report1)

print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:

<img width="832" alt="ex4 op1" src="https://user-images.githubusercontent.com/127508123/235941156-ff1c71d5-c4e6-4cb5-9d0c-17974050ca52.png">

<img width="730" alt="ex4 op2" src="https://user-images.githubusercontent.com/127508123/235941247-7affc757-1681-4f56-a4a2-314cc1cd2f45.png">

<img width="268" alt="ex4 op3" src="https://user-images.githubusercontent.com/127508123/235941289-9bf81571-5a48-4d9f-a258-5a28c3bc662a.png">

<img width="170" alt="ex4 op4" src="https://user-images.githubusercontent.com/127508123/235941346-11632c3f-21cf-4cb9-8aa4-4d00deac3257.png">

<img width="675" alt="ex4 op5" src="https://user-images.githubusercontent.com/127508123/235941388-60fd3e75-6197-4b57-a7c9-2f5c27da231b.png">

<img width="626" alt="ex4 op6" src="https://user-images.githubusercontent.com/127508123/235941472-72a6b53b-8e41-4f80-8576-0cee54e11a11.png">

<img width="372" alt="ex4 op7" src="https://user-images.githubusercontent.com/127508123/235941514-7e3dd87a-e67b-4e5a-842b-51843362da48.png">

<img width="560" alt="ex4 op8" src="https://user-images.githubusercontent.com/127508123/235941574-c925901e-565b-4806-8750-ac726e1663b8.png">

<img width="218" alt="ex4 op9" src="https://user-images.githubusercontent.com/127508123/235941611-98c55918-79e1-44cf-9d41-dcdd86116188.png">

<img width="268" alt="ex4 op10" src="https://user-images.githubusercontent.com/127508123/235941656-6ab6be7f-54e4-4806-8b45-b3f844d0abfb.png">

<img width="405" alt="ex4 op11" src="https://user-images.githubusercontent.com/127508123/235941687-71e24845-96ba-4d34-a00c-77e97ed213c0.png">

<img width="871" alt="ex4 op12" src="https://user-images.githubusercontent.com/127508123/235941718-ca27c258-689d-4cac-b65b-aa9333963166.png">


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
