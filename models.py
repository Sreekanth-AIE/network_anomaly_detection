import numpy as np
import pandas as pd
from tqdm import tqdm
# from config import list_of_models
import utils
from time import perf_counter

from utils import *

import os, joblib
# for model evaluation
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,precision_score,recall_score

def experimentation(X_train,X_test,y_train,y_test,list_of_models,gen_report=True,save_address=None,metrics_monitor="F1-Score",save_best=True,metric_balance_avg="weighted"):
    results = {"Algorithm":[],"Accuracy":[],"Precision":[],"Recall":[],"F1-Score":[],"Area-u-curve":[],"Training_time":[],"Testing_time":[]}
    for model_name, model in tqdm(list_of_models,total=len(list_of_models)):
        
        s1 = perf_counter()
        model.fit(X_train,y_train)
        tr_tm = perf_counter() - s1

        s2 = perf_counter()
        y_pred = model.predict(X_test)
        ts_tm = perf_counter() - s2

        results["Algorithm"].append(model_name)
        results["Accuracy"].append(accuracy_score(y_test,y_pred))
        results["Precision"].append(precision_score(y_test,y_pred,average=metric_balance_avg))
        results["Recall"].append(recall_score(y_test,y_pred,average=metric_balance_avg))
        results["F1-Score"].append(f1_score(y_test,y_pred,average=metric_balance_avg))
        results["Area-u-curve"].append(roc_auc_score(y_test,y_pred,average=metric_balance_avg))
        results["Training_time"].append(tr_tm)
        results["Testing_time"].append(ts_tm)
    
    results_df = pd.DataFrame(results)#,index=results["Algorithm"])
    results_df.sort_values([metrics_monitor],ascending=False,inplace=True,ignore_index=True)

    if save_best:
        best_model_name, best_model = list_of_models[list_of_models[:,0]==results_df.iloc[0,0]][0]
        print(f"best model: {best_model_name}")
        try:
            joblib.dump(best_model,save_address)
        except Exception as e:
            print("saved model file location is not acceptable")
            joblib.dump(best_model,f"{best_model_name}.h5")

        if gen_report:
            gen_test_report(results_df)
            return results_df,best_model_name, best_model
        else:
            return results_df,best_model_name, best_model
    
    else:
        if gen_report:
            gen_test_report(results_df)
            return results_df
        else:
            return results_df


   
        




# def sup_modelling(X_train,X_test,y_train,y_test):
#     results = {"Algorithm":[],"Accuracy":[],"F1-Score":[],"Area-u-curve":[],"Training_time":[],"Testing_time":[]}
#     for model_name, model, model_params in tqdm(list_of_models,total=len(list_of_models)):
#         temp_model = model(**model_params)
#         s1 = perf_counter()
#         temp_model.fit(X_train,y_train)
#         tr_tm = perf_counter() - s1

#         s2 = perf_counter()
#         y_pred = temp_model.predict(X_test)
#         ts_tm = perf_counter() - s2

#         results["Algorithm"].append(model_name)
#         results["Accuracy"].append(accuracy_score(y_test,y_pred))
#         results["F1-Score"].append(f1_score(y_test,y_pred,average="weighted"))
#         results["Area-u-curve"].append(roc_auc_score(y_test,y_pred,average="weighted"))
#         results["Training_time"].append(tr_tm)
#         results["Testing_time"].append(ts_tm)
    
#     report = pd.DataFrame(results)#,index=results["Algorithm"])
#     report.sort_values(["F1-Score"],ascending=False,inplace=True)

#     best_model_name = report.iloc[0,0]
#     print(f"Best Performing Model:  {best_model_name}")
#     best_model, bm_params = list_of_models[list_of_models[:,0] == best_model_name][1:]
#     y_pred = best_model.predict(X_test)
    
    
    
#     return report

#  *****************************************************************************************


#  from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score
# from sklearn import model_selection
# from tqdm import tqdm

# random_state = 0
# metric_balance_factor = "weighted"
# scoring = ['accuracy', f'precision_{metric_balance_factor}', f'recall_{metric_balance_factor}', f'f1_{metric_balance_factor}', f'roc_auc_ovr_{metric_balance_factor}']
# monitoring_index = 3
# monitoring_metrics = f"test_{scoring[monitoring_index]}" 

# list_of_models=np.array([["Decision Tree",DecisionTreeClassifier,{"random_state":random_state}],
#                          ["Random Forest",RandomForestClassifier,{"random_state":random_state}], 
#                          ["Extra Trees",ExtraTreesClassifier,{"random_state":random_state}],
#                          ["Gradient Boosting",GradientBoostingClassifier,{"random_state":random_state}], 
#                          ["Support Vector Machine",SVC,{"random_state":random_state}], 
#                          ["GaussianNB",GaussianNB,{}], 
#                          ["KNeighborsClassifier",KNeighborsClassifier,{"n_neighbors":7}], 
#                          ["LogisticRegression",LogisticRegression,{}]])

# kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

# eval_results = []
# test_results = []
# for model_name, model, model_params in tqdm(list_of_models,total=len(list_of_models)):
#         temp_model = model(**model_params)
#         cv_results = model_selection.cross_validate(temp_model, X_train_trans, y_train, cv=kfold, scoring=scoring)
#         temp_results_df = pd.DataFrame(cv_results)
#         temp_results_df["Algorithm"] = model_name
#         eval_results.append(temp_results_df)

# eval_results_df = pd.concat(eval_results,ignore_index=True)
# eval_results_df.sort_values([monitoring_metrics],ascending=False,inplace=True)

# best_model_name = eval_results_df.iloc[0,-1]
# # print(f"Best Performing Model:  {best_model_name}")
# # print(len(list_of_models[list_of_models[:,0]==best_model_name][0,1:]))
# best_model, bm_params = list_of_models[list_of_models[:,0] == best_model_name][0,1:]
# best_model = best_model(**bm_params)
# best_model.fit(X_train_trans,y_train)
# y_pred = best_model.predict(X_test_trans)

# test_results_df = pd.DataFrame({"metrics":["Accuracy","Precision","Recall","F1-Score","Area-u-curve"],
#                                 "Values":[accuracy_score(y_test,y_pred),
#                                           precision_score(y_test,y_pred,average=metric_balance_factor),
#                                           recall_score(y_test,y_pred,average=metric_balance_factor),
#                                           f1_score(y_test,y_pred,average=metric_balance_factor),
#                                           roc_auc_score(y_test,y_pred,average=metric_balance_factor)]})       




#  *****************************************************************************************

# from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_auc_score
# from sklearn import model_selection
# from tqdm import tqdm

# random_state = 0
# metric_balance_factor = "weighted"
# scoring = ['accuracy', f'precision_{metric_balance_factor}', f'recall_{metric_balance_factor}', f'f1_{metric_balance_factor}', f'roc_auc_ovr_{metric_balance_factor}']
# monitoring_index = 3
# monitoring_metrics = f"test_{scoring[monitoring_index]}" 

# list_of_models=np.array([["Decision Tree",DecisionTreeClassifier,{"random_state":random_state}],
#                          ["Random Forest",RandomForestClassifier,{"random_state":random_state}]])

# kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)

# eval_results = []
# test_results = {"Accuracy":[],"Precision":[],"Recall":[],"F1-Score":[],"Area-u-curve":[],"Algorithm":[]}
# for model_name, model, model_params in tqdm(list_of_models,total=len(list_of_models)):
#         temp_model = model(**model_params)
#         cv_results = model_selection.cross_validate(temp_model, X_train_trans, y_train, cv=kfold, scoring=scoring)
#         temp_results_df = pd.DataFrame(cv_results)
#         temp_results_df["Algorithm"] = model_name
#         eval_results.append(temp_results_df)

# eval_results_df = pd.concat(eval_results,ignore_index=True)
# eval_results_df.sort_values([monitoring_metrics],ascending=False,inplace=True)

# best_model_name = eval_results_df.iloc[0,-1]
# # print(f"Best Performing Model:  {best_model_name}")
# # print(len(list_of_models[list_of_models[:,0]==best_model_name][0,1:]))
# best_model, bm_params = list_of_models[list_of_models[:,0] == best_model_name][0,1:]
# best_model = best_model(**bm_params)
# best_model.fit(X_train_trans,y_train)
# y_pred = best_model.predict(X_test_trans)
# test_results["Algorithm"] = best_model_name
# test_results["Accuracy"].append(accuracy_score(y_test,y_pred))
# test_results["F1-Score"].append(f1_score(y_test,y_pred,average=metric_balance_factor))
# test_results["Precision"].append(precision_score(y_test,y_pred,average=metric_balance_factor))
# test_results["Recall"].append(recall_score(y_test,y_pred,average=metric_balance_factor))
# test_results["Area-u-curve"].append(roc_auc_score(y_test,y_pred,average=metric_balance_factor))
        
# test_results_df = pd.DataFrame(test_results)
# test_results_df