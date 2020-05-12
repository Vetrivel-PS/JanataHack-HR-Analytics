#!/usr/bin/env python
# coding: utf-8

# In[17]:


#importing libraries
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split


# In[274]:


from sklearn.svm import SVC
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score


# In[80]:


import plotly.express as px
from scipy import stats


# In[169]:


import math


# In[218]:


import category_encoders as ce


# In[3]:


#importing csv files in our workspace
data=pd.read_csv(r'C:\Users\krishang\Desktop\ANALYTICS_HR\train_jqd04QH.csv')
test_data=pd.read_csv(r'C:\Users\krishang\Desktop\ANALYTICS_HR\test_KaymcHn.csv')


# In[6]:


#making copy of our original data file
data_copy=data.copy()
test_data_copy=test_data.copy()


# In[61]:


data['last_new_job'].value_counts()


# In[74]:


data.describe()


# In[44]:


X_train.describe(include='all')


# In[62]:


#Splitting data into training and cv set
#train_test_split() is a method from sklearn library
X_train,X_test,y_train,y_test = train_test_split(data_copy.iloc[:,0:-1],data_copy.iloc[:,-1],test_size=0.3,random_state=0)


# In[63]:


X_train


# In[64]:


X_train['city'].value_counts()


# In[65]:


np.size(X_train,0)


# In[66]:


X_train['enrolled_university'].value_counts()


# In[67]:


X_train['education_level'].value_counts()


# In[71]:


X_train['last_new_job'].value_counts()


# In[72]:


X_train['company_type'].value_counts()


# In[49]:


X_train.dtypes


# In[69]:





# In[70]:


X_train['last_new_job'].replace(np.nan,'1',inplace=True)
X_train['last_new_job'].replace('never','0',inplace=True)
X_train['last_new_job'].replace('>4','5',inplace=True)
X_train['lat_new_job']=X_train['last_new_job'].astype('int')


# In[79]:


fig=px.histogram(X_train,x='company_type',y=X_train['company_type'],color=y_train)
fig.show()


# In[86]:


#pearson_coef,p_value=stats.pearsonr(X_train['company_type'],y_train)
stats.spearmanr(y_train,X_train['company_type'])


# In[ ]:





# In[98]:


X_train_copy=X_train.copy()
X_train_copy = pd.get_dummies(X_train, columns=['company_type'], prefix = ['company_type'])

print(X_train_copy.head())


# In[104]:


X_train_copy.head(10)


# In[138]:


X_train_copy.describe(include='all')#BEFORE REPLACING VALUES


# In[157]:


X_train_copy['experience'].replace(np.nan,'>20',inplace=True)
X_train_copy['major_discipline'].replace(np.nan,'STEM',inplace=True)
X_train_copy['education_level'].replace(np.nan,'Gradute',inplace=True)
X_train_copy['enrolled_university'].replace(np.nan,'no_enrollment',inplace=True)
X_train_copy = pd.get_dummies(X_train, columns=['company_type'], prefix = ['company_type'])
X_train_copy['company_size'].replace(np.NaN,'50-99',inplace=True)


# In[128]:


X_train_copy['company_size'].replace(np.NaN,'50-99',inplace=True)


# In[154]:


X_train['enrolled_university'].value_counts()


# In[158]:


#AFTER REPLACING VALUES
X_train_copy.describe(include='all')


# In[162]:


fig=px.histogram(X_train_copy,x='experience',y=X_train_copy['experience'],color=y_train)
fig.show()


# In[163]:


X_train_copy['experience'].value_counts()


# In[164]:


X_train_copy['experience'].replace('<1','0',inplace=True)
X_train_copy['experience'].replace('>20','21',inplace=True)


# In[183]:


fig1=px.histogram(X_train_copy,x='experience',y=X_train_copy['experience'],color=y_train)
fig1.show()


# In[203]:



X_train_copy = pd.get_dummies(X_train_copy, columns=['major_discipline'], prefix = ['major_discipline'])

print(X_train_copy.head())


# In[206]:



X_train_copy = pd.get_dummies(X_train_copy, columns=['enrolled_university'], prefix = ['enrolled_university'])
X_train_copy = pd.get_dummies(X_train_copy, columns=['education_level'], prefix = ['education_level'])


# In[208]:


X_train_copy['relevent_experience'].replace('Has relevent experience',1,inplace=True)
X_train_copy['relevent_experience'].replace('No relevent experience',0,inplace=True)


# In[205]:


X_train_copy['relevent_experience'].value_counts()


# In[209]:


X_train_copy.describe(include='all')


# In[215]:


fig3=px.histogram(X_train_copy,x='gender',y=X_train_copy['gender'],color=y_train)
fig3.show()
#AS PROB OF JOB REQUIREMENT DESPITE OF ANY GWNDER IS SAME HENCE WE CAN DROP COLOMN OF GENDER


# In[216]:


fig1=px.histogram(X_train_copy,x='company_size',y=X_train_copy['company_size'],color=y_train)
fig1.show()


# In[226]:


X_train_copy.drop(axis=1,columns=['gender','last_new_job'],inplace=True)


# In[222]:




encoder = ce.BinaryEncoder(cols=['city'])
X_train_copy= encoder.fit_transform(X_train_copy)
X_train_copy.head()


# In[223]:


X_train_copy.describe(include='all')


# In[231]:


#X_train_copy.dtypes


# In[228]:


encoder1 = ce.BinaryEncoder(cols=['experience'])
X_train_copy= encoder1.fit_transform(X_train_copy)

encoder2 = ce.BinaryEncoder(cols=['company_size'])
X_train_copy= encoder2.fit_transform(X_train_copy)


# In[232]:


X_train_copy.describe()


# In[233]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from statistics import mode


# In[293]:


X_train_copy.head()


# In[340]:


sum(y_pred)


# In[341]:


y_train.value_counts()


# In[338]:


model = LogisticRegression(max_iter = 1000)
model.fit(X_train_copy_fit, y_train)
y_pred = model.predict(X_train_copy_fit)
#accuracy = model.score(X_train_copy.iloc[:,1:3], y_test)
#print(accuracy)


# In[337]:


from sklearn.preprocessing import StandardScaler
X_train_copy_fit=StandardScaler().fit_transform(X_train_copy)


# In[339]:


y_pred


# In[244]:


np.size(X_train_copy,1)


# In[255]:


X_train_copy.dtypes


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[256]:


np.size(X_train_copy['training_hours'])


# In[310]:


precision = precision_score(y_train,y_pred)
recall = recall_score(y_train,y_pred)
accuracy = accuracy_score(y_train,y_pred)
f1 = f1_score(y_train,y_pred)


# In[311]:


accuracy


# In[290]:


(np.size(y_train)-sum(y_train))/np.size(y_train)


# In[291]:


sklearn.metrics.roc_auc_score(y_train, y_pred)


# In[300]:


import sklearn.metrics as metrics
metrics.roc_auc_score(y_train, y_pred)


# In[336]:


stats.spearmanr(y_train,X_train_copy.iloc[:,39])


# In[342]:


from sklearn.ensemble import RandomForestClassifier


# In[358]:


classifier=RandomForestClassifier(n_jobs=2,oob_score=2,n_estimators=500)
classifier.fit(X_train_copy,y_train)


# In[359]:


y_pred=classifier.predict(X_train_copy)


# In[363]:





# In[360]:


sum(y_pred)


# In[350]:


sum(y_train)


# In[351]:


precision = precision_score(y_train,y_pred)
recall = recall_score(y_train,y_pred)
accuracy = accuracy_score(y_train,y_pred)
f1 = f1_score(y_train,y_pred)


# In[354]:


metrics.roc_auc_score(y_train, y_pred)


# In[387]:


X_test_copy=X_test.copy()


# In[388]:


test_data_copy=test_data.copy()


# In[395]:


X_test_copy['last_new_job'].replace(np.nan,'1',inplace=True)
X_test_copy['last_new_job'].replace('never','0',inplace=True)
X_test_copy['last_new_job'].replace('>4','5',inplace=True)
X_test_copy['last_new_job']=X_test_copy['last_new_job'].astype('int')


# In[394]:


test_data_copy['last_new_job'].value_counts()


# In[391]:


X_test_copy['experience'].replace('<1','0',inplace=True)
X_test_copy['experience'].replace('>20','21',inplace=True)
X_test_copy['experience'].replace(np.nan,'>20',inplace=True)
X_test_copy['major_discipline'].replace(np.nan,'STEM',inplace=True)
X_test_copy['education_level'].replace(np.nan,'Gradute',inplace=True)
X_test_copy['enrolled_university'].replace(np.nan,'no_enrollment',inplace=True)
X_test_copy = pd.get_dummies(X_test_copy, columns=['company_type'], prefix = ['company_type'])
X_test_copy['company_size'].replace(np.NaN,'50-99',inplace=True)

test_data_copy['experience'].replace('<1','0',inplace=True)
test_data_copy['experience'].replace('>20','21',inplace=True)
test_data_copy['experience'].replace(np.nan,'>20',inplace=True)
test_data_copy['major_discipline'].replace(np.nan,'STEM',inplace=True)
test_data_copy['education_level'].replace(np.nan,'Gradute',inplace=True)
test_data_copy['enrolled_university'].replace(np.nan,'no_enrollment',inplace=True)
test_data_copy = pd.get_dummies(test_data_copy, columns=['company_type'], prefix = ['company_type'])
test_data_copy['company_size'].replace(np.NaN,'50-99',inplace=True)


# In[399]:


X_test_copy.describe(include='all')


# In[400]:


test_data_copy.describe(include='all')


# In[398]:


X_test_copy.drop(columns=['gender'],inplace=True)
test_data_copy.drop(columns=['gender'],inplace=True)


# In[401]:


X_test_copy = pd.get_dummies(X_test_copy, columns=['enrolled_university'], prefix = ['enrolled_university'])
X_test_copy = pd.get_dummies(X_test_copy, columns=['education_level'], prefix = ['education_level'])
encoder1 = ce.BinaryEncoder(cols=['experience'])
X_test_copy= encoder1.fit_transform(X_test_copy)

encoder2 = ce.BinaryEncoder(cols=['company_size'])
X_test_copy= encoder2.fit_transform(X_test_copy)
X_test_copy['relevent_experience'].replace('Has relevent experience',1,inplace=True)
X_test_copy['relevent_experience'].replace('No relevent experience',0,inplace=True)
encoder = ce.BinaryEncoder(cols=['city'])
X_test_copy= encoder.fit_transform(X_test_copy)
X_test_copy = pd.get_dummies(X_test_copy, columns=['major_discipline'], prefix = ['major_discipline'])



test_data_copy = pd.get_dummies(test_data_copy, columns=['enrolled_university'], prefix = ['enrolled_university'])
test_data_copy = pd.get_dummies(test_data_copy, columns=['education_level'], prefix = ['education_level'])
encoder1 = ce.BinaryEncoder(cols=['experience'])
test_data_copy= encoder1.fit_transform(test_data_copy)

encoder2 = ce.BinaryEncoder(cols=['company_size'])
test_data_copy= encoder2.fit_transform(test_data_copy)
test_data_copy['relevent_experience'].replace('Has relevent experience',1,inplace=True)
test_data_copy['relevent_experience'].replace('No relevent experience',0,inplace=True)
encoder = ce.BinaryEncoder(cols=['city'])
test_data_copy= encoder.fit_transform(test_data_copy)
test_data_copy = pd.get_dummies(test_data_copy, columns=['major_discipline'], prefix = ['major_discipline'])


# In[402]:


X_test_copy


# In[403]:


test_data_copy


# In[405]:


y_pred2=classifier.predict(X_test_copy)


# In[406]:


sum(y_pred2)


# In[ ]:


classifier=RandomForestClassifier(n_jobs=2,oob_score=2,n_estimators=500)
classifier.fit(X_train_copy,y_train)


# In[416]:


classifierL=LogisticRegression(solver='saga',max_iter=10000)
classifierL.fit(X_train_copy,y_train)


# In[424]:


y_pred=classifierL.predict(X_train_copy)


# In[425]:


y_pred2=classifierL.predict(X_test_copy)


# In[426]:


metrics.roc_auc_score(y_train, y_pred)


# In[427]:


metrics.roc_auc_score(y_test, y_pred2)


# In[428]:


y_pred


# In[429]:


y_pred2


# In[434]:


X_train.describe(include='all')


# In[436]:


import seaborn as sns; sns.set(color_codes=True)

ax = sns.regplot(x="city_development_index", y=y_train, data=X_train_copy)


# In[437]:


ax = sns.regplot(x="training_hours", y=y_train, data=X_train_copy)


# In[440]:


ax = sns.regplot(x="lat_new_job", y=y_train, data=X_train_copy)


# In[443]:


ax = sns.regplot(x="enrolled_university_no_enrollment", y=y_train, data=X_train_copy)


# In[442]:


X_train_copy


# In[444]:


ax = sns.regplot(x="enrolled_university_Full time course", y=y_train, data=X_train_copy)


# In[445]:


ax = sns.regplot(x="enrolled_university_Part time course", y=y_train, data=X_train_copy)


# In[447]:


ax = sns.regplot(x="education_level_Primary School", y=y_train, data=X_train_copy)


# In[448]:


ax = sns.regplot(x="education_level_Phd", y=y_train, data=X_train_copy)


# In[449]:


ax = sns.regplot(x="education_level_Masters", y=y_train, data=X_train_copy)


# In[450]:


ax = sns.regplot(x="education_level_High School", y=y_train, data=X_train_copy)


# In[451]:


ax = sns.regplot(x="education_level_Graduate", y=y_train, data=X_train_copy)


# In[460]:


ax = sns.regplot(x="city_6", y=y_train, data=X_train_copy)


# In[461]:


ax = sns.regplot(x="city_7", y=y_train, data=X_train_copy)


# In[463]:


ax = sns.regplot(x="relevent_experience", y=y_train, data=X_train_copy)


# In[473]:


ax = sns.regplot(x="company_size_1", y=y_train, data=X_train_copy)


# In[478]:


ax = sns.regplot(x="company_size_3", y=y_train, data=X_train_copy)


# In[482]:


ax = sns.regplot(x="company_type_Funded Startup", y=y_train, data=X_train_copy)


# In[485]:


ax = sns.regplot(x="company_type_Public Sector", y=y_train, data=X_train_copy)


# In[486]:


ax = sns.regplot(x="company_type_Pvt Ltd", y=y_train, data=X_train_copy)


# In[491]:


x = sns.regplot(x="major_discipline_STEM", y=y_train, data=X_train_copy)


# In[506]:


X_train_edit=X_train_copy['city_development_index']


# In[507]:


X_train_edit.head()


# In[543]:


X_train_edit = pd.DataFrame(X_train_copy,columns=['city_development_index','education_level_Primary School','enrolled_university_Full time course','relevent_experience','company_type_Pvt Ltd'])
X_train_edit


# In[544]:


X_test_edit = pd.DataFrame(X_test_copy,columns=['city_development_index','education_level_Primary School','enrolled_university_Full time course','relevent_experience','company_type_Pvt Ltd'])

test_data_edit = pd.DataFrame(test_data_copy,columns=['city_development_index','education_level_Primary School','enrolled_university_Full time course','relevent_experience','company_type_Pvt Ltd'])


# In[559]:


classifierL=LogisticRegression(class_weight='balanced',max_iter=1000)
classifierL.fit(X_train_edit,y_train)


# In[560]:


y_pred=classifierL.predict_proba(X_test_edit)[:,1]


# In[570]:


classifierR=RandomForestClassifier(n_jobs=4,oob_score=2,n_estimators=500,class_weight='balanced')
classifierR.fit(X_train_edit,y_train)


# In[571]:


classifierR.predict_proba(X_test_edit)[:,1]


# In[572]:


y_predR=classifierR.predict_proba(X_test_edit)[:,1]
metrics.roc_auc_score(y_test, y_predR)


# In[576]:


from sklearn.svm import SVC
classifierSVM = SVC(kernel='linear',probability=True)
classifierSVM.fit(X_train_edit, y_train)


# In[584]:


y_predSVM=classifierSVM.predict_proba(X_test_edit)[:,1]


# In[585]:


y_predSVM


# In[586]:


metrics.roc_auc_score(y_test, y_predSVM)


# In[ ]:





# In[622]:


from sklearn.tree import DecisionTreeClassifier
classifierDTC=DecisionTreeClassifier(random_state=0)
classifierDTC.fit(X_train_edit,y_train)
y_predDTC=classifierDTC.predict_proba(X_test_edit)[:,1]
metrics.roc_auc_score(y_test, y_predDTC)


# In[620]:


from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
classifierP=Perceptron()
classifierP.fit(X_train_edit,y_train)
y_predP=classifierP.predict(X_test_edit)
metrics.roc_auc_score(y_test, y_predP)


# In[618]:


from sklearn.neighbors import KNeighborsClassifier
classifierKNN=KNeighborsClassifier(n_neighbors=3)
classifierKNN.fit(X_train_edit,y_train)
y_predKNN=classifierKNN.predict_proba(X_test_edit)[:,1]
metrics.roc_auc_score(y_test, y_predKNN)


# In[616]:


classifierSV = SVC(kernel = 'rbf', random_state = 0,probability=True)
classifierSV.fit(X_train_edit, y_train)
y_predSV=classifierSV.predict_proba(X_test_edit)[:,1]
metrics.roc_auc_score(y_test, y_predSV)


# In[610]:


from sklearn.linear_model import LinearRegression
classifierLinearReg=LinearRegression()
classifierLinearReg.fit(X_train_edit, y_train)
y_predLinearReg=classifierLinearReg.predict(X_test_edit)
metrics.roc_auc_score(y_test, y_predLinearReg)


# In[624]:


from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(X_train_edit,y_train)
y_predgbk=gbk.predict_proba(X_test_edit)[:,1]
metrics.roc_auc_score(y_test, y_predgbk)


# In[563]:


y_pred


# In[517]:


classifierL.coef_


# In[548]:


metrics.roc_auc_score(y_test, y_pred)#simple lineat regression excluding city6


# In[564]:


metrics.roc_auc_score(y_test, y_pred)


# In[626]:


y_result=gbk.predict_proba(test_data_edit)[:,1]


# In[627]:


y_result


# In[628]:


df_submission=pd.read_csv(r'C:\Users\krishang\Downloads\sample_submission_sxfcbdx.csv')
df_submission['target']=y_result


# In[629]:


df_submission


# In[630]:


df_submission.to_csv(r'C:\Users\krishang\Desktop\ANALYTICS_HR\sample_submission.csv')


# In[ ]:
