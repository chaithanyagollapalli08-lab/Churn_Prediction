import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import sys
import logging_code
from logging_code import setup_logging
logger=setup_logging("var_out")
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import yeojohnson

def vt_outliers(X_train_num,X_test_num):
    try:
        logger.info(f"Before Train Column Name : {X_train_num.columns}")
        logger.info(f"Before Test Column Name : {X_test_num.columns}")
        qt = QuantileTransformer(output_distribution='normal', random_state=42)

        for i in X_train_num.columns:

            if X_train_num[i].nunique() == 2:
                continue

            elif i in ['MonthlyCharges', 'TotalChargesreplaced']:
                X_train_num[i + '_qt'] = qt.fit_transform(X_train_num[[i]])
                X_test_num[i + '_qt'] = qt.transform(X_test_num[[i]])

                X_train_num.drop(i, axis=1, inplace=True)
                X_test_num.drop(i, axis=1, inplace=True)

            elif i == 'tenure':
                X_train_num[i + '_yeo'], lam_val = yeojohnson(X_train_num[i])
                X_test_num[i + '_yeo'] = yeojohnson(X_test_num[i], lmbda=lam_val)

                X_train_num.drop(i, axis=1, inplace=True)
                X_test_num.drop(i, axis=1, inplace=True)

        logger.info(f"After Train Column Name : {X_train_num.columns}")
        logger.info(f"AFter Test Column Name : {X_test_num.columns}")
        #Trimming
        logger.info('###################################################')
        logger.info(f"Before Train Column Name : {X_train_num.columns}")
        logger.info(f"Before Test Column Name : {X_test_num.columns}")

        # ---- Initialize Quantile Transformer ----
        qt = QuantileTransformer(output_distribution='normal', random_state=42)

        # ---- Loop through columns ----
        for col in X_train_num.columns:

            # Skip binary columns
            if X_train_num[col].nunique() == 2:
                continue

            # Quantile transformation for MonthlyCharges and TotalChargesreplaced
            elif col in ['MonthlyCharges', 'TotalChargesreplaced']:
                X_train_num[col + '_qt'] = qt.fit_transform(X_train_num[[col]])
                X_test_num[col + '_qt'] = qt.transform(X_test_num[[col]])
                X_train_num.drop(col, axis=1, inplace=True)
                X_test_num.drop(col, axis=1, inplace=True)

            # Yeo-Johnson for tenure
            elif col == 'tenure':
                X_train_num[col + '_yeo'], lam_val = yeojohnson(X_train_num[col])
                X_test_num[col + '_yeo'] = yeojohnson(X_test_num[col], lmbda=lam_val)
                X_train_num.drop(col, axis=1, inplace=True)
                X_test_num.drop(col, axis=1, inplace=True)


        # ---- Trimming only tenure_yeo column ----
        col_trim = 'tenure_yeo'
        iqr = X_train_num[col_trim].quantile(0.75) - X_train_num[col_trim].quantile(0.25)
        upper_limit = X_train_num[col_trim].quantile(0.75) + 1.5 * iqr
        lower_limit = X_train_num[col_trim].quantile(0.25) - 1.5 * iqr

        # Use np.clip for clean trimming
        X_train_num['tenure_trim'] = np.clip(X_train_num[col_trim], lower_limit, upper_limit)
        X_test_num['tenure_trim'] = np.clip(X_test_num[col_trim], lower_limit, upper_limit)

        # Drop the temporary Yeo-Johnson column
        X_train_num.drop([col_trim], axis=1, inplace=True)
        X_test_num.drop([col_trim], axis=1, inplace=True)

        logger.info(f"After Trimming Train Columns: {X_train_num.columns}")
        logger.info(f"After Trimming Test Columns: {X_test_num.columns}")
        return X_train_num, X_test_num
    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")