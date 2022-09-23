# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:43:33 2022

@author: gbournigal
"""

import pandas as pd
import numpy as np
from utils import load_housing_data, HOUSING_PATH, full_pipeline, num_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import loguniform
from sklearn.feature_selection import SelectFromModel


def get_housing_prepared(full_pipeline):
    housing = load_housing_data(HOUSING_PATH)
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
        
    housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.select_dtypes(include=[np.number])
    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    
    full_pipeline = full_pipeline(num_pipeline, 
                                  OneHotEncoder(handle_unknown="ignore"), 
                                  num_attribs,
                                  cat_attribs)
    
    housing_prepared = full_pipeline.fit_transform(housing)
    
    return housing, housing_prepared, housing_labels, strat_test_set, full_pipeline


housing, housing_prepared, housing_labels, strat_test_set, full_pipeline = get_housing_prepared(full_pipeline)


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

# class TopFeatureSelector(BaseEstimator, TransformerMixin):
#     def __init__(self, feature_importances, k):
#         self.feature_importances = feature_importances
#         self.k = k
#     def fit(self, X, y=None):
#         self.feature_indices_ = np.argsort(self.feature_importances)[-self.k:]
#         print(self.feature_indices_)
#         return self
#     def transform(self, X):
#         print(X.shape)
#         return X[:, self.feature_indices_]

def exercise_3():
    forest_reg = RandomForestRegressor(n_estimators=180,
                                       max_features=7,
                                       random_state=42)
    
    
    preparation_selection_pipeline = Pipeline([
        ('preparation', full_pipeline),
        ('feature_selection', SelectFromModel(estimator=forest_reg,
                                              threshold=0.005))
    ])
    
    housing_prepared_top = preparation_selection_pipeline.fit_transform(housing, housing_labels)
    
    return preparation_selection_pipeline, housing_prepared_top, feature_importances


preparation_selection_pipeline, _, feature_importances = exercise_3()


def exercise_4():
    rnd_search = exercise_2()

    forest_reg = RandomForestRegressor(n_estimators=180,
                                       max_features=7,
                                       random_state=42)


    prep_select_predict_pipeline = Pipeline(
        [
        ('preparation', full_pipeline),
        ('feature_selection', SelectFromModel(estimator=forest_reg,
                                              threshold=0.005)),
        ('model', SVR(**rnd_search.best_params_))]
        )
    
    prep_select_predict_pipeline.fit(housing, housing_labels)
    return prep_select_predict_pipeline
    
prep_select_predict_pipeline = exercise_4()


def exercise_5(prep_select_predict_pipeline):
    prep_select_predict_pipeline.named_steps['preparation'].named_transformers_["cat"].handle_unknown = 'ignore'

    param_grid = [{
        'preparation__num__imputer__strategy': ['mean', 
                                                # 'median', 
                                                'most_frequent'],
        'feature_selection__threshold': [0.0001,
                                         0.001, 
                                         0.005, 
                                         0.01, 
                                         0.02, 
                                         0.05, 
                                         0.1, 
                                         0.2]
    }]
    
    grid_search_prep = GridSearchCV(prep_select_predict_pipeline, 
                                    param_grid,
                                    cv=2,
                                    scoring='neg_mean_squared_error', 
                                    verbose=2,
                                    error_score='raise',
                                    n_jobs=-1
                                    )
    grid_search_prep.fit(housing, housing_labels)
    
    grid_search_prep.best_params_
    negative_mse = grid_search_prep.best_score_
    rmse = np.sqrt(-negative_mse)
    print(rmse)
    

if __name__ == '__main__':
    exercise_5(prep_select_predict_pipeline)
