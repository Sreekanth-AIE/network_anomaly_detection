import numpy as np
import pandas as pd

# for modelling
from sklearn.tree import DecisionTreeClassifier # ExtraTreeClassifier can also be taken from sklearn.tree
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,IsolationForest
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 0

normal_models=np.array([["Decision Tree",DecisionTreeClassifier,{"random_state":RANDOM_STATE}],
                         ["Random Forest",RandomForestClassifier,{"random_state":RANDOM_STATE}], 
                         ["Extra Trees",ExtraTreesClassifier,{"random_state":RANDOM_STATE}],
                         ["Gradient Boosting",GradientBoostingClassifier,{"random_state":RANDOM_STATE}], 
                         ["Support Vector Machine",SVC,{"random_state":RANDOM_STATE}], 
                         ["GaussianNB",GaussianNB,{}], 
                         ["KNeighborsClassifier",KNeighborsClassifier,{"n_neighbors":7}], 
                         ["LogisticRegression",LogisticRegression,{}]])


peculiar_models = [[OneClassSVM],[IsolationForest],[LocalOutlierFactor]]
 
