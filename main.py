import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
#mean_squared_error(y_true, y_pred, squared=False)
import pandas as pd
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
import xgboost
import numpy as np
import sklearn
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from imblearn.under_sampling import RandomUnderSampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import hstack
from imblearn.under_sampling import RandomUnderSampler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col].astype(str))
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

#test commit 2 2
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train = MultiColumnLabelEncoder(columns = ['MSSubClass','Alley','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence','MiscFeature','PoolQC','SaleType','SaleCondition']).fit_transform(df_train)
df_test = MultiColumnLabelEncoder(columns = ['MSSubClass','Alley','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence','MiscFeature','PoolQC','SaleType','SaleCondition']).fit_transform(df_test)

####################### Outliers Removal Manually ############################
# Deleting outliers
df_train = df_train.drop(df_train[(df_train['GrLivArea'] > 4000) & (df_train['SalePrice'] < 300000)].index)
df_train = df_train.drop(df_train[(df_train['LotFrontage'] > 250) & (df_train['SalePrice'] < 400000)].index)
df_train = df_train.drop(df_train[(df_train['LotArea'] > 150000) & (df_train['SalePrice'] < 500000)].index)
df_train = df_train.drop(df_train[(df_train['OverallCond'] == 6) & (df_train['SalePrice'] > 550000)].index)
df_train = df_train.drop(df_train[(df_train['OverallCond'] == 2) & (df_train['SalePrice'] > 300000)].index)
df_train = df_train.drop(df_train[(df_train['YearBuilt'] < 1920) & (df_train['SalePrice'] > 300000)].index)
df_train = df_train.drop(df_train[(df_train['YearBuilt'] > 1990) & (df_train['SalePrice'] > 650000)].index)
df_train = df_train.drop(df_train[(df_train['Exterior1st'] == 6) & (df_train['SalePrice'] > 500000)].index)
df_train = df_train.drop(df_train[(df_train['MasVnrArea'] >1400) & (df_train['SalePrice'] > 200000)].index)
df_train = df_train.drop(df_train[(df_train['BsmtFinType2'] == 0) & (df_train['SalePrice'] > 500000)].index)
df_train = df_train.drop(df_train[(df_train['BsmtFinType1'] == 0) & (df_train['SalePrice'] > 500000)].index)
df_train = df_train.drop(df_train[(df_train['EnclosedPorch'] > 500) & (df_train['SalePrice'] < 300000)].index)

# ColumnsToCheck = list(df_train.columns)[50:-1]
# #ColumnsToCheck = ['GrLivArea']
# for col in ColumnsToCheck:
#     fig, ax = plt.subplots()
#     ax.scatter(x = df_train[col], y = df_train['SalePrice'])
#     plt.ylabel('SalePrice', fontsize=13)
#     plt.xlabel(col, fontsize=13)
#     plt.show()
#     print(str(col) + ' is clean!')
#     print(" ")
#######################################################################################################################################



X_train, X_val, Y_train, Y_val = train_test_split(df_train.iloc[:,1:-1], df_train.iloc[:,-1], test_size=0.25, shuffle=True)


best_model = lgb.LGBMRegressor(num_leaves=110, max_depth=40)
model = BaggingRegressor(base_estimator=best_model, n_estimators=10)

RMSE_list = []
MAPE_list = []

for i in range(10):
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(Y_val, y_pred, squared=False)
    mape = mean_absolute_percentage_error(Y_val, y_pred)
    RMSE_list.append(rmse)
    MAPE_list.append(mape)
print('RMSE: ' + str(round(np.mean(RMSE_list),3)) + ' | MAPE: ' + str(round(np.mean(MAPE_list),3)))



print("db")