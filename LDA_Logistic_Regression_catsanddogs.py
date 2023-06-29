# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 19:39:04 2023

@author: Parth
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
#import the data
dog = pd.read_csv("C:\CODING\DataScience\dog.csv")
cat = pd.read_csv("C:\CODING\DataScience\cat.csv")


#transpose the dataset
dog = dog.T
cat = cat.T
print(dog.shape)

plt.imshow(dog.iloc[0].values.reshape(64,64).T,cmap ="gray")

total = pd.concat([dog,cat])
# scale the data using min max scaler
minmax= MinMaxScaler()
total = minmax.fit_transform(total)
total = pd.DataFrame(total)
print(total.head())
lda = LinearDiscriminantAnalysis()
total['index']=0

#asssing 1 for dog and 0 for cat

total['index'][:80]=1
total['index'][80:]=0

x=total.drop('index',1)#here we are deleting the column
y = total['index']

#transform the data using LDA

transformed_data = lda.fit_transform(x,y)
print(transformed_data.shape)#here the number of features is 1 because we have 2 classes in the dataset


x_train,x_test, y_train ,y_test=train_test_split(x,y,test_size = 0.2,random_state=42)
#we use random state = 42 so that we dont get varied accuracy as it 
#ensures the training data and testing data segregatin remains same


#lets train the LDA model
#for Lda Y should be single dimesnsional
y_test = y_test.values.ravel()
y_train = y_train.values.ravel()
lda_model= lda.fit(x_train,y_train)
y_pred = lda_model.predict(x_test)

confusion_matrix(y_test, y_pred)
print(f"The accuracy of the model  using LDA is ,{format(accuracy_score(y_test, y_pred))}")
    

#lets train using logistic regression
log =LogisticRegression()
log_model=log.fit(x_train,y_train)
lda_predict= lda_model.predict(x_test)

print(f"The accuracy using LogisticsRegression is {format(accuracy_score(y_test,lda_predict))}")