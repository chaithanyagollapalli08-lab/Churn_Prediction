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
logger=setup_logging("filter_methods")
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import pearsonr
def fm(X_train_num,X_test_num,y_train,y_test):
    try:
        # Before
        logger.info(f'before X_train_num : {X_train_num.shape} \n : {X_train_num.columns}')
        logger.info(f'before X_test_num : {X_test_num.shape} \n : {X_test_num.columns}')

        # Apply Mutual Information on TRAIN only
        mi = mutual_info_classif(X_train_num, y_train, random_state=42)

        # Create DataFrame for scores
        mi_scores = pd.DataFrame({
            'Feature': X_train_num.columns,
            'MI Score': mi
        }).sort_values(by='MI Score', ascending=False)

        logger.info(f"\nMutual Information Scores:\n{mi_scores}")

        # Set threshold (you can tune this)
        threshold = 0.01

        # Select good features
        good_features = mi_scores[mi_scores['MI Score'] > threshold]['Feature']
        bad_features = mi_scores[mi_scores['MI Score'] <= threshold]['Feature']

        logger.info(f"Number of good columns are : {len(good_features)} \n : {list(good_features)}")
        logger.info(f"Number of bad columns are : {len(bad_features)} \n : {list(bad_features)}")

        # Drop bad features from both train and test
        X_train_num = X_train_num.drop(columns=bad_features, errors='ignore')
        X_test_num = X_test_num.drop(columns=bad_features, errors='ignore')

        # After
        logger.info(f'After X_train_num : {X_train_num.shape} \n : {X_train_num.columns}')
        logger.info(f'After X_test_num : {X_test_num.shape} \n : {X_test_num.columns}')

        logger.info(" HYPOTHESIS TESTING")
        c = []
        for i in X_train_num.columns:
             res = pearsonr(X_train_num[i], y_train)
             c.append(res)
        t = np.array(c)
        #print(t)
        p_value = pd.Series(t[:, 1], index=X_train_num.columns)
        ind = 0
        f = []
        for i in p_value:
             if i < 0.05:
                 f.append(X_train_num.columns[ind])
             ind = ind + 1
        logger.info(X_train_num.columns)
        logger.info(f)
        logger.info(f'After X_train_num : {X_train_num.shape} \n : {X_train_num.columns}')
        logger.info(f'After X_test_num : {X_test_num.shape} \n : {X_test_num.columns}')
        return X_train_num, X_test_num

    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_msg}  ')