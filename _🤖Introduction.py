# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 11:36:16 2022

@author: gbournigal
"""

import streamlit as st


st.set_page_config(
     page_title="""Hands-on Machine Learning""",
     page_icon="🤖",
     layout="wide",
 )

st.title("""🤖 Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow""")

st.write("""
         The following web app contains the solutions of multiple exercises of 
         the book Hands-On Machine Learninf with Scikit-Learn, Keras, and
         TensorFlow in the form of mini-projects. 
         
         The github project will contain a folder for each chapter with the
         jupyter notebooks or auxiliary code for the development of this web app.
         
         If the exercise is simple it will be done right on this streamlit, if it
         is more complex it will be redirected to another streamlit app just for that
         mini-project.
         """)