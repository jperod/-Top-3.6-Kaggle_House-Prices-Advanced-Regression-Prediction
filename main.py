import sns as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from scipy import stats
from scipy.stats import norm, skew #for some statistics
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
#mean_squared_error(y_true, y_pred, squared=False)
import pandas as pd
from progressbar import progressbar
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
import statsmodels.api as sm
from scipy import stats

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

# X_train, X_val, Y_train, Y_val = train_test_split(df_train.iloc[:, 1:-1], df_train.iloc[:, -1], test_size=0.25,
#                                                       shuffle=True)
# df_train = pd.concat([X_train , pd.DataFrame(Y_train)], axis=1)
# df_val = pd.concat([X_val, pd.DataFrame(Y_val)], axis=1)

####################### Outliers Removal Manually ############################
if True:
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

###################### Adding total sqfootage feature | Skewed features ###################################
if False:
    # Adding total sqfootage feature
    df_train['TotalSF'] = df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']
    df_test['TotalSF'] = df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']
    numeric_feats_train = df_train.dtypes[df_train.dtypes != "object"].index
    numeric_feats_val = df_test.dtypes[df_test.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats_train = df_train[numeric_feats_train].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewed_feats_val = df_test[numeric_feats_val].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness_train = pd.DataFrame({'Skew' :skewed_feats_train})
    skewness_train = skewness_train[abs(skewness_train) > 0.75]
    skewness_val= pd.DataFrame({'Skew' :skewed_feats_val})
    skewness_val = skewness_val[abs(skewness_val) > 0.75]

    from scipy.special import boxcox1p

    skewed_features_train = skewness_train.index
    skewed_features_val = skewness_val.index

    lam = 0.15
    for feat in skewed_features_train:
        # all_data[feat] += 1
        df_train[feat] = boxcox1p(df_train[feat], lam)
    for feat in skewed_features_val:
        # all_data[feat] += 1
        df_test[feat] = boxcox1p(df_test[feat], lam)

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
RMSE_list = []
MAPE_list = []

# X_train, Y_train = df_train.iloc[:,0:-1], df_train.iloc[:,-1]

for i in progressbar(range(10)):
    X_train, X_val, Y_train, Y_val = train_test_split(df_train.iloc[:, 1:-1], df_train.iloc[:, -1], test_size=0.25, shuffle=True)
    ###################### Log-transformation of the target variable #############################################
    if True:
        # Remove Outliers
        all_train = pd.concat([X_train,Y_train],axis=1)
        all_train = all_train.drop(all_train[(all_train['GrLivArea'] > 4000) & (all_train['SalePrice'] < 300000)].index)
        all_train = all_train.drop(all_train[(all_train['LotFrontage'] > 250) & (all_train['SalePrice'] < 400000)].index)
        all_train = all_train.drop(all_train[(all_train['LotArea'] > 150000) & (all_train['SalePrice'] < 500000)].index)
        all_train = all_train.drop(all_train[(all_train['OverallCond'] == 6) & (all_train['SalePrice'] > 550000)].index)
        all_train = all_train.drop(all_train[(all_train['OverallCond'] == 2) & (all_train['SalePrice'] > 300000)].index)
        all_train = all_train.drop(all_train[(all_train['YearBuilt'] < 1920) & (all_train['SalePrice'] > 300000)].index)
        all_train = all_train.drop(all_train[(all_train['YearBuilt'] > 1990) & (all_train['SalePrice'] > 650000)].index)
        all_train = all_train.drop(all_train[(all_train['Exterior1st'] == 6) & (all_train['SalePrice'] > 500000)].index)
        all_train = all_train.drop(all_train[(all_train['MasVnrArea'] > 1400) & (all_train['SalePrice'] > 200000)].index)
        all_train = all_train.drop(all_train[(all_train['BsmtFinType2'] == 0) & (all_train['SalePrice'] > 500000)].index)
        all_train = all_train.drop(all_train[(all_train['BsmtFinType1'] == 0) & (all_train['SalePrice'] > 500000)].index)
        all_train = all_train.drop(all_train[(all_train['EnclosedPorch'] > 500) & (all_train['SalePrice'] < 300000)].index)
        # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
        Y_train = np.log1p(Y_train)


    model_lgb.fit(X_train, Y_train)
    y_pred = np.expm1(model_lgb.predict(X_val))
    rmse = mean_squared_error(Y_val, y_pred, squared=False)
    mape = mean_absolute_percentage_error(Y_val, y_pred)
    RMSE_list.append(rmse)
    MAPE_list.append(mape)
print('RMSE: ' + str(round(np.mean(RMSE_list),3)) + ' | MAPE: ' + str(round(np.mean(MAPE_list),3)))

print("db")