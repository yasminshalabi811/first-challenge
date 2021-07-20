#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


db= pd.read_csv('diabetes.csv')


# In[6]:


db.head()


# In[7]:


db['Glucose'].replace(0,np.nan,inplace=True)
db['Glucose'].replace(np.nan, data['Glucose'].mean(),inplace=True)
db['BloodPressure'].replace(0,np.nan,inplace=True)
db['BloodPressure'].replace(np.nan, data['BloodPressure'].mean(),inplace=True)
db['SkinThickness'].replace(0,np.nan,inplace=True)
db['SkinThickness'].replace(np.nan, data['SkinThickness'].mean(),inplace=True)
db['Insulin'].replace(0,np.nan,inplace=True)
db['Insulin'].replace(np.nan, data['Insulin'].mean(),inplace=True)
db['BMI'].replace(0,np.nan,inplace=True)
db['BMI'].replace(np.nan, data['BMI'].mean(),inplace=True)


# In[8]:


db


# In[14]:


bins = [18, 40, 80]
labels = ['young', 'old']
db['AgeCat']=pd.cut(x=db['Age'], bins=bins, labels=labels)
db


# In[85]:


c= db.corr()
c


# In[96]:



sns.heatmap(c, annot=True)


# In[97]:


sns.pairplot(db,hue="AgeCat")


# In[94]:


sns.pairplot(db, vars=["BMI","Insulin","Outcome"],hue="AgeCat")


# In[226]:


db=db.dropna()
db.head()


# In[224]:


db.info()


# In[228]:


AgeCatColsDummy=pd.get_dummies(db["AgeCat"])


# In[230]:


db.head()


# In[234]:


db=pd.concat((db,AgeCatColsDummy),axis=1)


# In[235]:


db.head()


# In[247]:



x=db.values
y=db["Outcome"].values


y = db.Outcome
X = db.drop(['Outcome'], axis=1)

X.head()


# In[268]:


from sklearn.model_selection import train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, test_size=0.25)


# In[269]:


X_train.head()


# In[270]:



from sklearn import ensemble
from sklearn import metrics 

RFC_model = ensemble.RandomForestClassifier(n_estimators=100)

RFC_model.fit(X_train_full, y_train)
RFC_preds = DT_model.predict(X_valid_full)

acc= metrics.accuracy_score(RFC_preds , y_valid) 
acc


# In[271]:


from sklearn.tree import DecisionTreeClassifier as dtree
DT_model=dtree()

from sklearn import metrics


DT_model.fit(X_train_full, y_train)
DT_preds = DT_model.predict(X_valid_full)
acc= metrics .accuracy_score(y_valid,DT_preds)
acc


# In[272]:



from sklearn.svm import SVC
from sklearn import metrics

SVC_model = SVC(kernel = 'rbf')


SVC_model.fit(X_train_full, y_train)
SVC_preds = SVC_model.predict(X_valid_full)

acc= metrics.accuracy_score(y_valid, SVC_preds)
acc


# In[ ]:





# In[ ]:




