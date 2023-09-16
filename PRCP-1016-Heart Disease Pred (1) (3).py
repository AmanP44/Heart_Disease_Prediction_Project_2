#!/usr/bin/env python
# coding: utf-8

# # PRCP-1016-HeartDisease Pred
# 
# ## PTID-CDS-JAN-23-1448
# 
# ## Project Name: Heart Disease Prediction
# ### Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.
# 
# ### Understanding the Problem Statement in this Business Case :
# Day by day the cases of heart diseases are increasing at a rapid rate and it’s very Important and concerning to predict any such diseases beforehand. This diagnosis is a difficult task i.e. it should be performed precisely and efficiently.We need a heart disease prediction system to predict whether the patient is likely to be diagnosed with a heart disease or not using the medical history of the patient.

# # Domain Analysis
# **patient_id** = ID of particular patient
# 
# **slope_of_peak_exercise_st_segment** = While a high ST depression is considered normal & healthy. The “ slope ” hue, refers to the peak exercise ST
# 
# **thal** = A blood disorder called thalassemia,[normal, reversible defect, fixed defect]
# 
# **resting_blood_pressure** = blood pressure tells a lot about your general health. High blood pressure or hypertension can lead to several heart related
# 
# **chest_pain_type** = Most of the chest pain causes are not dangerous to health, but some are serious, while the least cases are life-threatening.
# 
# **num_major_vessels** = Major Blood Vessels of the Heart: Blood exits the right ventricle through the pulmonary trunk artery. Approximately two inches
# 
# **fasting_blood_sugar_gt_120_mg_per_dl** = (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false
# 
# **resting_ekg_results** = The electrocardiogram (ECG or EKG) is a test that measures the heart’s electrical activity, and a resting ECG is
# 
# **serum_cholesterol_mg_per_dl** = A person’s serum cholesterol level represents the amount of total cholesterol in their blood. A person’s serum
# 
# **oldpeak_eq_st_depression** = oldpeak = ST depression induced by exercise relative to rest, a measure of abnormality in electrocardiograms
# 
# **sex** = sex (1 = male; 0 = female)
# 
# **Age** = Age of patients
# 
# **max_heart_rate_achieved** = This is the average maximum number of times your heart should beat per minute during exercise.
# 
# **exercise_induced_angina** = Angina is chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood.[0: no, 1: yes]
# 
# **heart_disease_present** = Target variable 0:No heart disease, 1:heart disease

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# to visualise all the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[2]:


#Reading Values and labels dataset in a different DataFrames
data_val = pd.read_csv('values.csv')
data_lab = pd.read_csv('labels.csv')


# In[3]:


# Load Data
df1=pd.read_csv('values.csv')
df1


# In[4]:


df2=pd.read_csv('labels.csv')
df2


# In[5]:


# Renaming Common column name
df2.rename({'patient_id':'p_id'},axis=1,inplace=True)
df2


# In[6]:


# Combining  both Labels and Value dataframes together
data=pd.concat([df1,df2],axis=1)
data


# In[7]:


# Delete one of the common column
data=data.drop('p_id',axis=1)
data


# # Basic checks

# In[8]:


data.columns


# In[9]:


# we are going to rename some columns

data.rename({'slope_of_peak_exercise_st_segment':'slope_sts','resting_blood_pressure':'rest_bp',
              'fasting_blood_sugar_gt_120_mg_per_dl':'fbs','serum_cholesterol_mg_per_dl':'serum_cholesterol',
             'oldpeak_eq_st_depression':'st_depression','max_heart_rate_achieved':'max_hr',
              'exercise_induced_angina':'exe_angina','chest_pain_type':'cp_type','num_major_vessels':'num_mv',
            'resting_ekg_results':'rest_ekg','heart_disease_present':'Hd_Present'},axis=1,inplace=True)


# In[10]:


data.columns


# In[11]:


data.head() # it shows data of 5 first rows

# as you can see now after renaming columns , it looks more neat


# In[12]:


data.tail() # it shows data of last 5 rows


# In[13]:


data.shape


#  #  Insight
# - Data set has 15 columns and 180 rows
# - 14 parameters and 1 target columns

# In[14]:


data.dtypes


#  # Insight
# - 2 columns are categorical
# - 13 columns are numerical

# In[15]:


data.info() # Describing datatypes, entries and non-null values


# In[16]:


# we can conclude that there are no null values


# In[17]:


data.describe().T


#  # Insight
# - rest_bp- minimum value is 94,average value is 130 and maximum value is 180
# - serum_cholesterol-minimum value is 126,average value is 246 and maximum value is 564
# - max_hr-minimum value is 96,average value is 152 and maximum value is 202
# - age-minimum age is 29,average age is 55 and maximum age is 77

# In[18]:


# Finding Unique Values & Checking Value counts of target variable
data.Hd_Present.unique()


# In[19]:


data.Hd_Present.value_counts()


#  # Insight
# - 1- Heart Disease Present
# - 0- Heart Disease Abscent

# # EXPLORATORY DATA ANALYSIS

#  # Univariate

# In[20]:


data.columns


# In[21]:


sns.histplot(x='rest_bp',data=data)


# In[22]:


sns.histplot(x='serum_cholesterol',data=data)


# In[23]:


sns.distplot(data['max_hr'])


# In[24]:


sns.distplot(data['age'])


# In[25]:


sns.histplot(x='st_depression',data=data)


# In[26]:


#Categorical and Nominal data 


# In[27]:


data['slope_sts'].value_counts().plot(kind='pie',autopct='%.2f')


# In[28]:


sns.countplot(x='thal',data=data)


# In[29]:


sns.countplot(x='cp_type',data=data)


# In[30]:


sns.countplot(x='rest_ekg',data=data)


# In[31]:


sns.countplot(x='Hd_Present',data=data)


# ### now we will use boxplot to find out outliers present in numerical data.

# In[32]:


sns.boxplot(x='rest_bp',data=data)


# In[33]:


sns.boxplot(x='max_hr',data=data)


# In[34]:


sns.boxplot(x='serum_cholesterol',data=data)


# In[35]:


sns.boxplot(x='age',data=data)


# # insights of univariate analysis

# - Distribution of resting blood pressure is more betweem 110 to 140
# 
# - Distribution of cholestrol is more between 200 to 300
# 
# - Distribution of age is more between 50 to 70
# 
# - Distribution of maximum heart rate is more betweem 140 to 180
# 

# # outliers
# 
# - rest_bp and has serum_cholesterol

# # Bivariate

# In[36]:


# Analysing Heart Disease with respect to sex.
sns.countplot(data=data,x='sex',hue='Hd_Present')
plt.title('Sex v/s HD_Present')


# # Insight
# - 1:male , o:female
# 
# - Heart Disease is present more in males.

# In[37]:


# Analysing Thalassemia with respect to sex.
sns.countplot(data=data,x='sex',hue='thal')
plt.title('Sex v/s Thalassemia')


# # Insight
#  - 0: female, 1: male
#  - male have more thalassemia

# In[38]:


# Analysing Heart Disease with respect to Chestpain Type
sns.countplot(data=data,x='cp_type',hue='Hd_Present')
plt.title('Chestpain type v/s Heart Disease Present')


#  # Insight
# - 1 = typical angina
# - 2 = atypical angina
# - 3 = non — anginal pain
# - 4 = asymptotic
# - 1- Heart Disease Present and 0- Heart Disease Abscent
# - asymptomatic chest pain is more.

# In[39]:


# Analysing Heart Disease with respect to Thalassemia
sns.countplot(data=data,x='thal',hue='Hd_Present')
plt.title('Thalassemia v/s Heart Disease Present')


# # Insight
# - 1- Heart Disease Present and 0- Heart Disease Abscent
# - reversible defect is more

# In[40]:


# Analysing Heart Disease with respect to Slope of peak exercise
sns.countplot(data=data,x='slope_sts',hue='Hd_Present')
plt.title('Slope_of_peak_exercise v/s Heart Disease Present')


# # Insight
# - 1 = upsloping
# - 2 = flat
# - 3 = downsloping
# - 1- Heart Disease Present and 0- Heart Disease Abscent
# - Flat Slope is more

# # DATA PREPROCESSING AND FEATURE ENGINEERING

# In[41]:


# checking correlation
plt.figure(figsize=(20, 20))#canvas size
sns.heatmap(data.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":10})


# In[42]:


data.corr()


# In[43]:


#Checking null values
data.isnull().sum()


# # there is no missing values

# In[44]:


# Checking for duplicate values
data.duplicated().sum()


# # there is no missing values

# In[45]:


# as thal was a categorical data we have to change it to numerical
from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()

data.thal=lc.fit_transform(data.thal)


# In[46]:


data.head()


# In[47]:


# thal values have been changed


# # Checking and Handling Outliers
# 
# - there are two outliers rest_bp and serum_cholesterol
# 

# In[48]:


sns.boxplot(x='rest_bp',data=data)


# In[49]:


sns.boxplot(x='serum_cholesterol',data=data)


# In[50]:


# Droping Patient ID as its not significant
data.drop('patient_id',axis='columns',inplace=True)


# In[51]:


data.columns


# In[52]:


data_con=['slope_sts','thal','cp_type','num_mv','rest_ekg']


# In[53]:


data_cont=['serum_cholesterol','age','max_hr']


# In[54]:


#creating dummies or one hot encoder for nominal values
data=pd.get_dummies(data,columns=data_con,drop_first=True)


# In[55]:


data.head()


# In[56]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
data[data_cont]=sc.fit_transform(data[data_cont])
data.head()


# In[57]:


# Splitting data into X and Y


# In[59]:


x=data.drop('Hd_Present',axis=1)


# In[61]:


y=data['Hd_Present']


# In[62]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=45)


# ## Balancing the Data

# In[63]:


from collections import Counter# importing counter to check count of each label
from imblearn.over_sampling import SMOTE #for balancing the data
sm=SMOTE()#object creation
print(Counter(y))# checking count for each class
x_sm,y_sm=sm.fit_resample(x,y)#applying sampling on target variable
print(Counter(y_sm))# checking count after sampling for  each class


# In[64]:


## preparing training and testing data
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_sm, y_sm, test_size=0.25, random_state=42)


# ## Logistic Regression

# In[65]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[66]:


#Training Data
y_pred=model.predict(x_train)
y_pred


# In[67]:


y_pred_prob=model.predict_proba(x_train)


# In[68]:


from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,f1_score


# In[69]:


cm=confusion_matrix(y_train,y_pred)
cm


# In[70]:


recall=recall_score(y_train,y_pred)
recall


# In[71]:


precision=precision_score(y_train,y_pred)
precision


# In[72]:


ac1=accuracy_score(y_train,y_pred)
ac1


# In[73]:


f1=f1_score(y_train,y_pred)
f1


# In[74]:


cr=classification_report(y_train,y_pred)
print(cr)


# In[75]:


#Testing Data
y_pred1=model.predict(x_test)
y_pred1


# In[76]:


y_pred_prob1=model.predict_proba(x_test)


# In[77]:


cm=confusion_matrix(y_test,y_pred1)
cm


# In[78]:


recall=recall_score(y_test,y_pred1)
recall


# In[79]:


f1=f1_score(y_test,y_pred1)
f1


# In[80]:


acc=accuracy_score(y_test,y_pred1)
acc


# In[81]:


cr=classification_report(y_test,y_pred1)
print(cr)


# ## Decision Tree

# In[82]:


print(Counter(y_test))


# In[83]:


print(Counter(y_train))


# In[84]:


from sklearn.tree import DecisionTreeClassifier # importing decision tree classifie

dft=DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf= 1, min_samples_split= 3, splitter= 'random')#object creation for decision tree
dft.fit(x_train,y_train)#training the model
y_hat=dft.predict(x_test)#prediction
y_hat#predicted values


# In[85]:


## Evalauting the model
from sklearn.metrics import accuracy_score,classification_report,f1_score    #importing mertics to check model performance
##Training score
y_train_predict=dft.predict(x_train)#passing X_train to predict Y_train
acc_train=accuracy_score(y_train,y_train_predict)#checking accuracy
acc_train


# In[86]:


y_test_pred = dft.predict(x_test)


# In[87]:


acc_test = accuracy_score(y_test,y_test_pred)
acc_test


# ## Hyperparameter Tuning

# In[88]:


from sklearn.model_selection import GridSearchCV
params = {
    "criterion":("gini", "entropy"),
    "splitter":("best", "random"),
    "max_depth":(list(range(1, 20))),
    "min_samples_split":[2,3,4],
    "min_samples_leaf":list(range(1, 20)),
}


# In[89]:


tree_clf = DecisionTreeClassifier(random_state=3)
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
tree_cv.fit(x_train,y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")


# In[90]:


tree_cv.best_score_


# In[91]:


dft1=DecisionTreeClassifier(criterion='gini',max_depth=1,min_samples_leaf= 1,min_samples_split=2,splitter='best')
dft1.fit(x_train,y_train)#training the model
y_pred=dft1.predict(x_test)#prediction
y_pred#predicted values


# In[92]:


y_train_predict=dft1.predict(x_train)#predicting training data to check training performance
y_train_predict


# In[93]:


print(classification_report(y_train,y_train_predict))# it will give precision,recall,f1 scores and accuracy


# In[94]:


print(classification_report(y_test,y_test_pred))


# In[95]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.tree import plot_tree

plt.figure(figsize=(24,14))
plot_tree(dft, filled=True)


# ## Random Forest

# In[96]:


print(Counter(y_train))


# In[97]:


from sklearn.ensemble import RandomForestClassifier#importing randomforest

rf_clf = RandomForestClassifier(n_estimators=100)#object creation ,taking 100 decision tree in random forest
rf_clf.fit(x_train,y_train)#training the data


# In[98]:


y_p=rf_clf.predict(x_train)


# In[99]:


y_predict=rf_clf.predict(x_test)#testing
y_predict


# In[100]:


from sklearn.metrics import accuracy_score,classification_report,f1_score
print(classification_report(y_test,y_predict))


# In[101]:


acc=accuracy_score(y_test,y_predict)
acc


# In[102]:


f1_Score=f1_score(y_test,y_predict)
f1_Score


# In[103]:


recall=recall_score(y_test,y_predict)
recall


# In[104]:


precision=precision_score(y_test,y_predict)
precision


# ## SVM

# In[105]:


from sklearn.svm import SVC
svclassifier=SVC()#basemodel with default parameters
svclassifier.fit(x_train,y_train)


# In[106]:


#predict output for x_test
y_hat=svclassifier.predict(x_test)


# ## Hyperparameter Tunning

# In[107]:


from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,5,10,50,60,70],
           'gamma':[1,0.1,0.01,0.001,0.0001],
           'random_state':(list(range(1,20)))}# defining parameter range
model=SVC()
grid=GridSearchCV(model,param_grid,refit=True,verbose=2,scoring='f1',cv=5)
grid.fit(x,y)#fitting the model for grid search


# In[108]:


print(grid.best_params_)


# In[109]:


clf=SVC(C=10,gamma=0.001,random_state=1)


# In[110]:


clf.fit(x_train,y_train)


# In[111]:


y_clf=clf.predict(x_test)


# In[112]:


print(classification_report(y_test,y_clf))


# In[113]:


## checking cross validation score
from sklearn.model_selection import cross_val_score

scores_after = cross_val_score(clf,x,y,cv=3,scoring='f1')
print(scores_after)
print("Cross validation Score:",scores_after.mean())
print("Std :",scores_after.std())
#std of < 0.05 is good.


# ## KNN

# In[114]:


from sklearn.neighbors import KNeighborsClassifier


# In[115]:


## taking optimal k to determine how many nearest neighbors  to create

# create a list to store the error values for each k
error_rate = []

# Will take some time
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
    # if predicted value is not equal to actual value, returns true (which is taken as 1) else false(0).
    # Adds all the value and takes mean of it. So for each k-value, gets the mean of error.
    #print(np.mean(pred_i != y_test))


# In[116]:


error_rate


# In[117]:


plt.figure(figsize=(10,6))
plt.plot(range(1,11),error_rate,color='blue', linestyle='dashed',
         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[118]:


# let's fit the data into KNN model and see how well it performs:
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)# Predict
y_pred = knn.predict(x_test)


# In[119]:


# Checking Accuracy score
print("The accuracy score is : ", accuracy_score(y_test,y_pred))


# In[120]:


recall=recall_score(y_test,y_pred)
recall


# In[121]:


f1=f1_score(y_test,y_pred)
f1


# In[122]:


precision=precision_score(y_test,y_pred)
precision


# In[123]:


print(classification_report(y_test,y_pred))


# In[ ]:




