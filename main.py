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
logger=setup_logging("main")
import seaborn as sns
from var_out import vt_outliers
from Random_Sample_Imputation import handle_missing_values
from filter_methods import fm
from cat_to_num import c_t_n
from imblearn.over_sampling import SMOTE
from feature_scaling import fs

class CHURN:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(self.path)
            logger.info(f'total data size {self.df.shape}')
            self.df['SIM_provider']=np.random.choice(['Jio','Airtel','VI','BSNL'],size=len(self.df),p=[0.4,0.3,0.2,0.1])
            logger.info(f'total data size {self.df.shape}')
            # Convert TotalCharges to numeric
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            self.df = self.df.drop('customerID', axis=1)
            logger.info(f'total null values : {self.df.isnull().sum()}')
            logger.info(self.df.shape)
            self.X = self.df.drop('Churn', axis=1)
            self.y = self.df['Churn']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)

            self.y_train = self.y_train.str.strip().str.capitalize().map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            self.y_test = self.y_test.str.strip().str.capitalize().map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
            logger.info(f"X_train size: {len(self.X_train)}\n, y_train size: {len(self.y_train)}\n,total train size:{self.X_train.shape}")
            logger.info(f"X_test size: {len(self.X_test)}\n, y_test size: {len(self.y_test)}\n,total test size:{self.X_test.shape}")
        except Exception as e:
          err_type, err_msg, err_line = sys.exc_info()
          logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_msg} ')

    def missing_values(self):
        try:
            logger.info(f"Before Handling NUll values X_train Column names and shape : {self.X_train.shape} \n : {self.X_train.columns} : {self.X_train.isnull().sum()}")
            logger.info(f"Before Handling NUll values X_test Column names and shape : {self.X_test.shape} \n : {self.X_test.columns} : {self.X_test.isnull().sum()}")
            self.X_train['TotalCharges'] = pd.to_numeric(self.X_train['TotalCharges'],errors='coerce')
            self.X_test['TotalCharges'] = pd.to_numeric(self.X_test['TotalCharges'],errors='coerce')
            self.X_train, self.X_test = handle_missing_values(self.X_train, self.X_test)
            logger.info(f"After Handling NUll values X_train Column names and shape : {self.X_train.shape} \n : {self.X_train.columns} : {self.X_train.isnull().sum()}")
            logger.info(f"After Handling NUll values X_test Column names and shape : {self.X_test.shape} \n : {self.X_test.columns} : {self.X_test.isnull().sum()}")
        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_msg} ')

    def data_separation(self):
        try:
            self.X_train_num_cols = self.X_train.select_dtypes(exclude='object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude='object')
            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')

            logger.info(f"{self.X_train_num_cols.columns}: {self.X_train_num_cols.shape}")
            logger.info(f"{self.X_test_num_cols.columns} : {self.X_test_num_cols.shape}")
            logger.info('======================================================')
            logger.info(f"{self.X_train_cat_cols.columns} : {self.X_train_cat_cols.shape}")
            logger.info(f"{self.X_test_cat_cols.columns} : {self.X_test_cat_cols.shape}")
        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_msg} ')

    def var_transformation(self):
        try:
            #nd
            #plt.figure(figsize=(5,3))
            #for i in self.X_train_num_cols.columns:
             #   self.X_train_num_cols[i].plot(kind='kde',color='red',label=i)
              #  plt.legend()
               # plt.show()
            #outliers
            #plt.figure(figsize=(5,3))
            #for i in self.X_train_num_cols.columns:
             #   sns.boxplot(x=self.X_train_num_cols[i])
             #   plt.show()
            logger.info(f"Before Transformation Train Columns: {self.X_train_num_cols.columns}")
            logger.info(f"Before Transformation Test Columns: {self.X_test_num_cols.columns}")
            self.X_train_num_cols, self.X_test_num_cols = vt_outliers(self.X_train_num_cols, self.X_test_num_cols)
            logger.info(f"after Train Column Name : {self.X_train_num_cols.columns}")
            logger.info(f"after Test Column Name : {self.X_test_num_cols.columns}")

        except Exception as e:
            logger.error(f"Error in combine_data: {str(e)}")
            return self.X_train, self.X_test
    def feature_selection(self):
        try:
            self.X_train_num_cols, self.X_test_num_cols = fm(self.X_train_num_cols, self.X_test_num_cols, self.y_train,self.y_test)
        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_msg}  ')

    def categorical_to_num(self):
        try:
            self.X_train_cat_cols, self.X_test_cat_cols = c_t_n(self.X_train_cat_cols, self.X_test_cat_cols)
            self.X_train_num_cols.reset_index(drop=True, inplace=True)
            self.X_train_cat_cols.reset_index(drop=True, inplace=True)
            self.X_test_num_cols.reset_index(drop=True, inplace=True)
            self.X_test_cat_cols.reset_index(drop=True, inplace=True)
            self.training_data = pd.concat([self.X_train_num_cols, self.X_train_cat_cols], axis=1)
            self.testing_data = pd.concat([self.X_test_num_cols, self.X_test_cat_cols], axis=1)
            self.training_data.fillna(0, inplace=True)
            self.testing_data.fillna(0, inplace=True)

            logger.info(f'Training data columns : {self.training_data.columns} and its shape : {self.training_data.shape}')
            logger.info(f'testing data columns : {self.testing_data.columns} and its shape : {self.testing_data.shape}')
        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_type} and the error message is {err_msg}')

    def data_balancing(self):
        try:
            logger.info(f"Number of Rows for Good Customer {1} : {sum(self.y_train == 1)}")
            logger.info(f"Number of Rows for Bad Customer {0} : {sum(self.y_train == 0)}")
            logger.info(f"Training data size : {self.training_data.shape}")
            self.training_data.fillna(0, inplace=True)
            sm = SMOTE(random_state=42)

            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)

            logger.info(f"Number of Rows for Good Customer {1} : {sum(self.y_train_bal == 1)}")
            logger.info(f"Number of Rows for Bad Customer {0} : {sum(self.y_train_bal == 0)}")
            logger.info(f"Training data size : {self.training_data_bal.shape}")

            fs(self.training_data_bal, self.y_train_bal, self.testing_data, self.y_test)
        except Exception as e:
            er_type, er_msg, er_line = sys.exc_info()
            logger.info(f"Error in line no : {er_line.tb_lineno} due to : {er_msg}")

if __name__=="__main__":
  try:
    obj=CHURN('churn_data.csv')
    obj.missing_values()
    obj.data_separation()
    obj.var_transformation()
    obj.feature_selection()
    obj.categorical_to_num()
    obj.data_balancing()

  except Exception as e:
      err_type, err_msg, err_line = sys.exc_info()
      logger.info(f'Error in lineno {err_line.tb_lineno} and error is due to {err_msg} ' )