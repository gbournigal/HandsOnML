# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:43:33 2022

@author: gbournigal
"""

import pandas as pd
import numpy as np
from utils import load_housing_data, HOUSING_PATH, full_pipeline, num_pipeline
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from scipy.stats import loguniform


def get_housing_prepared(full_pipeline):
    housing = load_housing_data(HOUSING_PATH)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.select_dtypes(include=[np.number])
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    
    full_pipeline = full_pipeline(num_pipeline, 
                                     OneHotEncoder(), 
                                     num_attribs,
                                     cat_attribs)
    
    housing_prepared = full_pipeline.fit_transform(housing)
    
    return housing_prepared, housing_labels, strat_test_set


housing_prepared, housing_labels, strat_test_set = get_housing_prepared(full_pipeline)

def exercise_1():
    param_grid = [
        {'kernel': ['linear'], 
         'C': [0.1, 1, 10, 100, 1000]
         },
        {'kernel': ['rbf'], 
         'C': [0.1, 1, 10, 100, 1000], 
         'gamma': [0.01, 0.1, 0.3, 1, 5]
         },
      ]

    svr_reg = SVR()
    grid_search = GridSearchCV(svr_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1, 
                               verbose=2)
    grid_search.fit(housing_prepared, housing_labels)
    return grid_search
    

def exercise_2():
    param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': loguniform(100, 500000),
        'gamma': loguniform(0.01, 2)
    }
    
    svm_reg = SVR()
    rnd_search = RandomizedSearchCV(svm_reg, 
                                    param_distributions=param_distribs,
                                    n_iter=30,
                                    cv=5, 
                                    scoring='neg_mean_squared_error',
                                    verbose=2, 
                                    random_state=42,
                                    n_jobs=-1)
    rnd_search.fit(housing_prepared, housing_labels)
    return rnd_search
