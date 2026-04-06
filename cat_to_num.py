import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import logging
from logging_code import setup_logging
logger=setup_logging("cat_to_num")
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
def c_t_n(X_train_cat,X_test_cat):
    try:
        logger.info(f'**************  Nominal Encoder  ***************')

        logger.info(f'Before X_train_cat columns : {X_train_cat.columns} and its shape is {X_train_cat.shape}')
        logger.info(f'Before X_test_cat columns : {X_test_cat.columns} and its shape is {X_test_cat.shape}')

        # -------------------------------
        # 1. Binary Encoding
        # -------------------------------
        binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']

        for col in binary_cols:
            X_train_cat[col] = X_train_cat[col].map({'Yes': 1, 'No': 0,'No internet service':0,
            'No phone service':0})
            X_test_cat[col] = X_test_cat[col].map({'Yes': 1, 'No': 0,'No internet service':0,
            'No phone service':0})

        # Handle gender separately
        X_train_cat['gender'] = X_train_cat['gender'].map({'Male': 1, 'Female': 0})
        X_test_cat['gender'] = X_test_cat['gender'].map({'Male': 1, 'Female': 0})

        # -------------------------------
        # 2. One-Hot Encoding (only needed columns)
        # -------------------------------
        ohe_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'PaymentMethod'
        ]

        nom = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        nom.fit(X_train_cat[ohe_cols])

        train_val = nom.transform(X_train_cat[ohe_cols])
        test_val = nom.transform(X_test_cat[ohe_cols])

        t1 = pd.DataFrame(train_val, columns=nom.get_feature_names_out(ohe_cols))
        t2 = pd.DataFrame(test_val, columns=nom.get_feature_names_out(ohe_cols))

        # reset index
        X_train_cat.reset_index(drop=True, inplace=True)
        X_test_cat.reset_index(drop=True, inplace=True)
        t1.reset_index(drop=True, inplace=True)
        t2.reset_index(drop=True, inplace=True)

        # concat
        X_train_cat = pd.concat([X_train_cat, t1], axis=1)
        X_test_cat = pd.concat([X_test_cat, t2], axis=1)

        # drop original OHE columns
        X_train_cat.drop(ohe_cols, axis=1, inplace=True)
        X_test_cat.drop(ohe_cols, axis=1, inplace=True)

        # -------------------------------
        # 3. Frequency Encoding (SIM_provider)
        # -------------------------------
        freq_map = X_train_cat['SIM_provider'].value_counts()

        X_train_cat['SIM_provider'] = X_train_cat['SIM_provider'].map(freq_map)
        X_test_cat['SIM_provider'] = X_test_cat['SIM_provider'].map(freq_map)

        # handle unseen categories in test
        X_test_cat['SIM_provider'].fillna(0, inplace=True)

        # -------------------------------
        # Final Logs
        # -------------------------------
        logger.info(f'After X_train_cat columns : {X_train_cat.columns} and its shape is {X_train_cat.shape}')
        logger.info(f'After X_test_cat columns : {X_test_cat.columns} and its shape is {X_test_cat.shape}')

        logger.info(f'**************  Odinal Encoder  ***************')
        od = OrdinalEncoder()
        od.fit(X_train_cat[['Contract']])
        val_train = od.transform(X_train_cat[['Contract']])
        val_test = od.transform(X_test_cat[['Contract']])
        p1 = pd.DataFrame(val_train)
        p2 = pd.DataFrame(val_test)
        p1.columns = od.get_feature_names_out() + '_odinal'
        p2.columns = od.get_feature_names_out() + '_odinal'
        p1.reset_index(drop=True, inplace=True)
        p2.reset_index(drop=True, inplace=True)
        X_train_cat = pd.concat([X_train_cat, p1], axis=1)
        X_test_cat = pd.concat([X_test_cat, p2], axis=1)

        X_train_cat = X_train_cat.drop(['Contract'], axis=1)
        X_test_cat = X_test_cat.drop(['Contract'], axis=1)
        logger.info(f'After Odinal X_train_cat columns : {X_train_cat.columns} and its shape : {X_train_cat.shape}')
        logger.info(f'After Odinal X_test_cat columns : {X_test_cat.columns} and its shape : {X_test_cat.shape}')
        return X_train_cat, X_test_cat
    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_type} and the error message is {err_msg}')