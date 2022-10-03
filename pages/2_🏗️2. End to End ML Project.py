# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:47:04 2022

@author: gbournigal
"""

import pickle
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
    grid_search = pickle.load(open('/app/handsonml/chapter_2/ex_1.sav'))
    # grid_search = pickle.load(
    #     open(
    #         "C:/Users/gbournigal/Documents/GitHub/HandsOnML/chapter_2/ex_1.sav",
    #         "rb",
    #     )
    # )
    st.write(grid_search.best_params_)
