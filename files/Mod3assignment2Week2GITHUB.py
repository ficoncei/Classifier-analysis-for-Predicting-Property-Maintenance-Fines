

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
# from adspy_shared_utilities import plot_class_regions_for_classifier


#function that gets a df and a column name and plots histograms 
#for cases that complied or not against info in the df (column_name=info)
def plot_compliance(df,column_name):
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    df.loc[df['compliance']==1,column_name].hist(color='green')
    plt.title('Payed in time')
    plt.xlabel(column_name)
    plt.ylabel('# of tickets')
    plt.subplot(122)
    df.loc[df['compliance']==0,column_name].hist(color='red')
    plt.title("Not payed in time")
    plt.xlabel(column_name)
    plt.show()

def plot_scores(scores):
    x = np.arange(4)
    plt.figure(figsize=(10,6))
    plt.bar(x,scores)
    plt.ylim(0.7, 0.85)
    plt.title('Mean Cross Validation AUC Scores for tested Classifiers')
    plt.xticks(x,('Naive Bayes','Gradient boosting','Random Forest',',Tuned Random Forest'))
    plt.xlabel('Tested Classifiers')
    plt.ylabel('AUC score')

    
def plot_features_compliance(df):
    for name in df.columns:
        plot_compliance(df,name)

def plot_cumulative_explained_variance(X):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Features')
    plt.ylabel('Cumulative Explained Variance');
    
def plot_explained_variance_per_feature(X):
    pca = PCA().fit(X)
    pca_variance = pca.explained_variance_
    print(pca_variance)
    for i in range(5):
        plt.figure(figsize=(8, 6))
        plt.bar(np.arange(15-i),pca_variance[i:], alpha=0.5, align='center', label='Feature variance')
        plt.legend()
        plt.ylabel('Variance Ratio')
        plt.xlabel('# Features')
        plt.show()

def train_Random_Forest_Classifier(df,best_params):
    X = df.drop('compliance', axis=1)
    y = df.loc[:,'compliance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print('train size:', X_train.shape)
    print('test size: ', X_test.shape)
    if best_params==0:
        model = RandomForestClassifier().fit(X_train, y_train)
    if best_params!=0:
        model = RandomForestClassifier(n_estimators=best_params['n_estimators'],max_depth=best_params['max_depth']).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scored = roc_auc_score(y_test, y_pred)
    print('AUC score: ', scored)
    cv_score = cross_val_score(model, X_train, y_train, scoring='roc_auc')
    print('Cross validation AUC scores: ', cv_score)
    print('Mean of Cross validation AUC scores: ', cv_score.mean())
    return model,cv_score.mean()

def train_nb_Classifier(df):
    X = df.drop('compliance', axis=1)
    y = df.loc[:,'compliance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print('train size:', X_train.shape)
    print('test size: ', X_test.shape)
    nbclf = GaussianNB().fit(X_train, y_train)
    y_pred = nbclf.predict(X_test)
    scored = roc_auc_score(y_test, y_pred)
    print('AUC score: ', scored)
    cv_score = cross_val_score(nbclf, X_train, y_train, scoring='roc_auc')
    print('Cross validation AUC scores: ', cv_score)
    print('Mean of Cross validation AUC scores: ', cv_score.mean())
    return nbclf,cv_score.mean()

def train_GB_Classifier(df):
    X = df.drop('compliance', axis=1)
    y = df.loc[:,'compliance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    print('train size:', X_train.shape)
    print('test size: ', X_test.shape)
    clf = GradientBoostingClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    scored = roc_auc_score(y_test, y_pred)
    print('AUC score: ', scored)
    cv_score = cross_val_score(clf, X_train, y_train, scoring='roc_auc')
    print('Cross validation AUC scores: ', cv_score)
    print('Mean of Cross validation AUC scores: ', cv_score.mean())
    return clf,cv_score.mean()

def run_Rand_SearchCV(model,df):
    X = df.drop('compliance', axis=1)
    y = df.loc[:,'compliance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 110, num = 11)]
    
    # max depth
    max_depth = [int(x) for x in np.linspace(100, 110, num = 2)]
    max_depth.append(None)
    
    # create random grid
    random_grid = {
     'n_estimators': n_estimators,
     'max_depth': max_depth
     }
    # Random search of parameters
    model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 33, cv = 3, random_state=42, n_jobs = -1)
    # Fit the model
    model_random.fit(X_train, y_train)
    # print results
    print(model_random.best_params_)
    return model_random.best_params_
    

def apply_Random_Forest_Classifier(RF_model,X):
    y_pred = RF_model.predict(X)
    return y_pred
    

def load_clean_data():
        
    train_df = pd.read_csv('train.csv',encoding="ISO-8859-1")
    test_df = pd.read_csv('test.csv',encoding="ISO-8859-1")
    #addresses_df = pd.read_csv('addresses.csv',encoding="ISO-8859-1")
    #latlons = pd.read_csv('latlons.csv',encoding="ISO-8859-1")
    
    train_df.index = train_df['ticket_id']
    test_df.index = test_df['ticket_id']
    
#Eliminate blank values from the true y values as 
#the person has been found not responsible
    train_df['compliance'].replace('', np.nan, inplace=True)
    train_df.dropna(subset=['compliance'], inplace=True)
    
#convert string to datetime format:
    train_df['hearing_date']=pd.to_datetime(train_df['hearing_date'])
    train_df['ticket_issued_date']=pd.to_datetime(train_df['ticket_issued_date'])
    
    test_df['hearing_date']=pd.to_datetime(test_df['hearing_date'])
    test_df['ticket_issued_date']=pd.to_datetime(test_df['ticket_issued_date'])
    
#add a column with the number of days between ticket issue and hearing dates
    train_df['hearing_time_diff'] = (train_df['hearing_date']-train_df['ticket_issued_date']).dt.days
    train_df['hearing_time_diff'].fillna(value=0, inplace=True)
    train_df['hearing_time_diff'] = train_df['hearing_time_diff'].astype(int)
    
    test_df['hearing_time_diff'] = (test_df['hearing_date']-test_df['ticket_issued_date']).dt.days
    test_df['hearing_time_diff'].fillna(value=0, inplace=True)
    test_df['hearing_time_diff'] = test_df['hearing_time_diff'].astype(int)
    
#Drop columns with irrelevant, blank or redundant data
    
    proc_train_df=train_df.drop(['ticket_id','payment_amount','payment_date',
       'payment_status','balance_due','collection_status','violation_street_number',
       'violation_street_name','violation_zip_code', 'mailing_address_str_number',
       'mailing_address_str_name','non_us_str_code', 'country','city', 'state',
       'ticket_issued_date', 'hearing_date','violation_description', 'grafitti_status',
       'compliance_detail'],axis=1)

    test_df = test_df[['agency_name', 'inspector_name', 'violator_name', 'zip_code',
       'violation_code', 'disposition', 'fine_amount', 'admin_fee',
       'state_fee', 'late_fee', 'discount_amount', 'clean_up_cost',
       'judgment_amount','hearing_time_diff']]
    
#here it is noticed that some hearing dates occur before the ticket issuing date
#which indicates wrong data in the dataset. As it could be misleading, it is 
#better to remove them
#print("Total rows with inconsistent dates: ",np.sum(proc_train_df['hearing_time_diff'] < 0))
    proc_train_df = proc_train_df.drop(proc_train_df[proc_train_df['hearing_time_diff'] < 0].index)
    
    proc_train_df['fine_amount'].fillna(value=0, inplace=True)
    test_df['fine_amount'].fillna(value=0, inplace=True)
    
#Fill NAs in the rest of the df and replace strings with codes
    for col in proc_train_df.select_dtypes(include=['object']).columns:
        proc_train_df[col].fillna(value='None', inplace=True)
        proc_train_df[col] = pd.Categorical(proc_train_df[col], categories=proc_train_df[col].unique()).codes

    for col in test_df.select_dtypes(include=['object']).columns:
        test_df[col].fillna(value='None', inplace=True)
        test_df[col] = pd.Categorical(test_df[col], categories=test_df[col].unique()).codes

    
    return proc_train_df,test_df

proc_train_df,test_df=load_clean_data()
plot_features_compliance(proc_train_df)
random_forest_model,rf_score=train_Random_Forest_Classifier(proc_train_df,0)
y_pred=apply_Random_Forest_Classifier(random_forest_model,test_df)
best_params=run_Rand_SearchCV(random_forest_model,proc_train_df)
random_forest_model,rf_imp_score=train_Random_Forest_Classifier(proc_train_df,best_params)
naive_bayes_clf,nb_score=train_nb_Classifier(proc_train_df)
GB_clf,gb_score=train_GB_Classifier(proc_train_df)
scores=(nb_score,gb_score,rf_score,rf_imp_score)
plot_scores(scores)

#proc_train_df



