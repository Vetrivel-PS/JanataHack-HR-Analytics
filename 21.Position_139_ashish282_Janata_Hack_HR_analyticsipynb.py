import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score,precision_recall_fscore_support,precision_score,accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier




enrol_df=pd.read_csv('F://JanataHack/train_jqd04QH.csv')
print(enrol_df.head())
print(enrol_df.columns)

##enrol_df["target"].replace({1:"yes"}, inplace=True)
##enrol_df["target"].replace({0:"no"}, inplace=True)


##target_no=raw_enrol_df[raw_enrol_df.target==0]
##target_yes=raw_enrol_df[raw_enrol_df.target==1]
##
##
##df_minority_upsampled=resample(target_yes,replace=True,n_samples=15000,random_state=102)
##
##enrol_df=pd.concat([target_no,df_minority_upsampled])
##
##enrol_df=shuffle(enrol_df)

###'enrollee_id', 'city', 'city_development_index', 'gender',
##       #'relevent_experience', 'enrolled_university', 'education_level',
##       #'major_discipline', 'experience', 'company_size', 'company_type',
##       #'last_new_job', 'training_hours', 'target'
##
##
print('city_development_index',enrol_df['city_development_index'].isnull().sum())
print('gender',enrol_df['gender'].isnull().sum())
print('relevent_experience',enrol_df['relevent_experience'].isnull().sum())
print('enrolled_university',enrol_df['enrolled_university'].isnull().sum())
print('education_level',enrol_df['education_level'].isnull().sum())
print('major_discipline',enrol_df['major_discipline'].isnull().sum())
print('experience',enrol_df['experience'].isnull().sum())
print('company_size',enrol_df['company_size'].isnull().sum())
print('company_type',enrol_df['company_type'].isnull().sum())
print('last_new_job',enrol_df['last_new_job'].isnull().sum())
print('training_hours',enrol_df['training_hours'].isnull().sum())

enrol_df['gender'].fillna('Not_mentioned', inplace=True)
enrol_df['enrolled_university'].fillna('no_enrollment', inplace=True)
enrol_df['education_level'].fillna('not_entered', inplace=True)
enrol_df['major_discipline'].fillna('Other', inplace=True)
enrol_df['experience'].fillna(0, inplace=True)
enrol_df['company_size'].fillna("not_given", inplace=True)
enrol_df['company_type'].fillna("not_appl", inplace=True)
enrol_df['last_new_job'].fillna("never", inplace=True)



enrol_df["company_size"].replace({"10/49":"10-49"}, inplace=True)
enrol_df["experience"].replace({">20":int(21)}, inplace=True)
enrol_df["experience"].replace({"<1":int(0)}, inplace=True)


all_features=['city_development_index', 'gender',
       'relevent_experience', 'enrolled_university', 'education_level',
       'major_discipline', 'experience', 'company_size', 'company_type',
       'last_new_job', 'training_hours']


categorical_features=['gender','relevent_experience','enrolled_university', 'education_level','major_discipline',
                      'company_size', 'company_type','last_new_job']

enrol_encoded_df=pd.get_dummies(enrol_df[all_features],columns=categorical_features,drop_first=True)

enrol_encoded_df['experience']=enrol_encoded_df['experience'].astype(int)

def get_significant_vars(lm):
    var_p_vals_df=pd.DataFrame(lm.pvalues)
    var_p_vals_df['vars']=var_p_vals_df.index
    var_p_vals_df.columns=['pvals','vars']
    return list (var_p_vals_df[var_p_vals_df.pvals<=0.05]['vars'])



X=sm.add_constant(enrol_encoded_df)

y=enrol_df['target']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)


##clf.best_params_
##{'max_depth': 10, 'max_features': 'sqrt', 'n_estimators': 20}

randm_clf=RandomForestClassifier(max_depth=10,max_features='sqrt',n_estimators=20)

randm_clf.fit(X_train,y_train)
randm_clf.predict(X_test)

y_pred=randm_clf.predict(X_test)



threshold = 0.135320

predicted_proba = randm_clf.predict_proba(X_test)
predicted = (predicted_proba[:,1] >= threshold).astype('int')

y_pred_df=pd.DataFrame({"actual":y_test,
                        "predicted":predicted})


def draw_roc_curve(model,test_X,test_y):
    test_results_df=pd.DataFrame({'actual':test_y})
    test_results_df=test_results_df.reset_index()
    predict_proba_df=pd.DataFrame(model.predict_proba(test_X))
    test_results_df['chd_1']=predict_proba_df.iloc[:,1:2]

    fpr,tpr,thresholds=roc_curve(test_results_df.actual,test_results_df.chd_1,
                                         drop_intermediate=False)

    auc_score=roc_auc_score(test_results_df.actual,test_results_df.chd_1)

    plt.figure(figsize=(8,6))

    plt.plot(fpr,tpr,label='ROC curve (area=%0.2f)'%auc_score)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.legend(loc='lower right')
    plt.show()
    return auc_score,fpr,tpr,thresholds
    
    
auc_score,fpr,tpr,thresholds=draw_roc_curve(randm_clf,X_test,y_test)

tpr_fpr=pd.DataFrame({'tpr':tpr,'fpr':fpr,'thresholds':thresholds})






tpr_fpr['diff']=tpr_fpr.tpr-tpr_fpr.fpr

print(tpr_fpr.sort_values('diff',ascending=False)[0:5])

print("Accuracy:",accuracy_score(y_test, y_pred))
##
#y_pred_df['predicted_new']=y_pred_df.predicted_prob.map(lambda x:1 if x>0.3 else 0)
##
##
enrol_test_df=pd.read_csv('F://JanataHack/test_KaymcHn.csv')
##
enrol_test_df['gender'].fillna('Not_mentioned', inplace=True)
enrol_test_df['enrolled_university'].fillna('no_enrollment', inplace=True)
enrol_test_df['education_level'].fillna('not_entered', inplace=True)
enrol_test_df['major_discipline'].fillna('Other', inplace=True)
enrol_test_df['experience'].fillna(0, inplace=True)
enrol_test_df['company_size'].fillna("not_given", inplace=True)
enrol_test_df['company_type'].fillna("not_appl", inplace=True)
enrol_test_df['last_new_job'].fillna("never", inplace=True)
##
##
##
enrol_test_df["company_size"].replace({"10/49":"10-49"}, inplace=True)
enrol_test_df["experience"].replace({">20":int(21)}, inplace=True)
enrol_test_df["experience"].replace({"<1":int(0)}, inplace=True)
##
##
enrol_test_encoded_df=pd.get_dummies(enrol_test_df[all_features],columns=categorical_features,drop_first=True)

enrol_test_encoded_df['experience']=enrol_test_encoded_df['experience'].astype(int)

X1=sm.add_constant(enrol_test_encoded_df)




predicted_proba = randm_clf.predict_proba(X1)
predicted = (predicted_proba[:,1] >= threshold).astype('int')


y_test_pred_df=pd.DataFrame({"predicted":predicted})


  
output_df=pd.DataFrame(enrol_test_df.enrollee_id)
output_df['target']=y_test_pred_df['predicted']
output_df.to_csv(r"F:\\JanataHack\output.csv",index=False)

