# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:47:04 2022

@author: gbournigal
"""

import pickle
import pandas as pd
import streamlit as st

import os

cur_path = os.path.dirname(__file__)
new_path = os.path.relpath("..\\chapter_2", cur_path)

st.set_page_config(
    page_title="""2. End-to-End ML Project""",
    page_icon="🏗️",
    layout="wide",
)

st.title("🏗️ End-to-End Machine Learning Project")

with st.expander("Q 2.1:"):
    st.write(
        """Try a Support Vector Machine regressor (sklearn.svm.SVR), with various hyperparameters such as kernel="linear" (with various values for the C hyperparameter) or kernel="rbf" (with various values for the C and gamma hyperparameters). Don't worry about what these hyperparameters mean for now. How does the best SVR predictor perform?"""
    )
    
    st.write("The param grid used is:")
    param_grid =         [{"kernel": ["linear"], "C": [0.1, 1, 10, 100, 1000]},
            {
                "kernel": ["rbf"],
                "C": [0.1, 1, 10, 100, 1000],
                "gamma": [0.01, 0.1, 0.3, 1, 5],
            }]
    st.write(param_grid)
    grid_search = pickle.load(open('/app/handsonml/chapter_2/ex_1.sav', 'rb'))
    df_grid = pd.DataFrame(grid_search.cv_results_)[['mean_test_score',
                                                'param_C',
                                                'param_gamma',
                                                'param_kernel']].sort_values('mean_test_score', ascending=False)
    st.dataframe(df_grid.style.format(
                formatter={'mean_test_score': lambda x: "{:,.1f}".format(x/1000000),
                           'param_C': "{:,.1f}",
                           'param_gamma': '{:,.2f}'
                          }),
        use_container_width=True)
    st.write("The results indicates that the best parameters are C=1000 with linear kernel. Is substantially better than the worst combination of parameters.")
    st.write("Nevertheless, since C=1000 is the top of the range, it would be helpful to keep trying with higher values of C.")
    

with st.expander("Q 2.2:"):
    st.write(
        """Try replacing GridSearchCV with RandomizedSearchCV."""
    )
    st.write("The distributions used for randomized search are:")
    st.write("C: loguniform(100, 500000)")
    st.write("gamma: loguniform(0.01, 2)")
    rand_search = pickle.load(open('/app/handsonml/chapter_2/ex_2.sav', 'rb'))
    df_rand = pd.DataFrame(rand_search.cv_results_)[['mean_test_score',
                                                'param_C',
                                                'param_gamma',
                                                'param_kernel']].sort_values('mean_test_score', ascending=False)
    st.dataframe(df_rand.style.format(
                formatter={'mean_test_score': lambda x: "{:,.1f}".format(x/1000000),
                           'param_C': "{:,.1f}",
                           'param_gamma': '{:,.2f}'
                          }),
        use_container_width=True)
    st.write("There is an improvement from the GridSearch. Now the best parameters are: C=76,569.1, kernel=rbf, gamma=0.24")
    st.write("We could potentially restrict the distributions more after the results found.")
  

