import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
from logging_code import setup_logging
logger = setup_logging("Random_Sample_Imputation")

def handle_missing_values(X_train,X_test):
    try:
        logger.info(f"Before Handling NUll values X_train Column names and shape : {X_train.shape} \n : {X_train.columns} : {X_train.isnull().sum()}")
        logger.info(f"Before Handling NUll values X_test Column names and shape : {X_test.shape} \n : {X_test.columns} : {X_test.isnull().sum()}")

        for i in X_train.columns:
            if X_train[i].isnull().sum()>0:
                X_train[i+"replaced"]=X_train[i].copy()
                X_test[i+"replaced"] = X_test[i].copy()
                s=X_train[i].dropna().sample(X_train[i].isnull().sum(),random_state=42)
                s1 = X_test[i].dropna().sample(X_test[i].isnull().sum(), random_state=42)
                s.index=X_train[X_train[i].isnull()].index
                s1.index = X_test[X_test[i].isnull()].index
                X_train.loc[X_train[i].isnull(),i+'replaced']=s
                X_test.loc[X_test[i].isnull(),i+'replaced']=s1
                X_train = X_train.drop([i],axis=1)
                X_test = X_test.drop([i], axis=1)
        logger.info(f"After Handling NUll values X_train Column names and shape : {X_train.shape} \n : {X_train.columns} : {X_train.isnull().sum()}")
        logger.info(f"After Handling NUll values X_test Column names and shape : {X_test.shape} \n : {X_test.columns} : {X_test.isnull().sum()}")
        return X_train,X_test
    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f"Error in line no : {err_line.tb_lineno} due to : {er_msg}")