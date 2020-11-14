import warnings

from mlxtend.regressor import StackingCVRegressor
from scipy.special import boxcox1p
from sklearn.svm import SVR
import progressbar
warnings.filterwarnings("ignore")
from sklearn.linear_model import ElasticNet, Lasso, RidgeCV, LassoCV, ElasticNetCV, Ridge
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import skew, stats, shapiro, boxcox_normmax  # for some statistics
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
from sklearn.preprocessing import LabelEncoder
#mean_squared_error(y_true, y_pred, squared=False)
import pandas as pd


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
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

def TestModel(model, it, val_size,verbose):
    RMSE_list = []
    MAPE_list = []


    for i in range(it):
        X_train, X_val, Y_train, Y_val = train_test_split(df_train.drop(columns="SalePrice").iloc[:,1:], df_train["SalePrice"], test_size=val_size, shuffle=True)
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
            X_train = all_train.iloc[:,0:-1]; Y_train = all_train.iloc[:,-1];
            # We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
            Y_train = np.log1p(Y_train)
            ###################### Adding total sqfootage feature | Skewed features ###################################
            # Adding total sqfootage feature
            X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
            X_val['TotalSF'] = X_val['TotalBsmtSF'] + X_val['1stFlrSF'] + X_val['2ndFlrSF']
            # More Transformation
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            X_val = np.array(X_val)
            Y_val = np.array(Y_val)

        # print
        model.fit(X_train, Y_train)
        y_pred = np.expm1(model.predict(X_val))
        rmse = mean_squared_log_error(Y_val, y_pred)
        mape = mean_absolute_percentage_error(Y_val, y_pred)
        RMSE_list.append(rmse)
        MAPE_list.append(mape)
        print(i+1)

    if verbose:
        print('RMSLE: ' + str(round(np.mean(RMSE_list),3)) + ' | MAPE: ' + str(round(np.mean(MAPE_list),3)))

    # submission = pd.concat([df_test.iloc[:, 0], pd.DataFrame(y_pred, columns=['SalePrice'])], axis=1)

    return
def MakePrediction(model):

    X_train, Y_train = df_train.drop(columns=["SalePrice","Id"]), df_train["SalePrice"]
    Y_train = np.log1p(Y_train)
    X_test = df_test.drop(columns=["Id"])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    model.fit(X_train, Y_train)
    # y_pred = model.predict(X_test)
    y_pred = np.expm1(model.predict(X_test))

    submission = pd.concat([df_test.iloc[:,0],pd.DataFrame(y_pred, columns=['SalePrice'])],axis=1)

    return submission
#HPO Optimization
def TuneLGB():
    #Tune XGB hyperparameters:
    #28/48 | [5, -1, 0.05, 720, 40, 0.3, 19284.39, 7.96] | Best RMSE: 19284 => MAPE: 7.959

    num_leaves = [5,6]
    max_depth = [-1]
    learning_rate = [0.05]
    n_estimators = [700,750,800]
    max_bin = [40]
    colsample_bytree = [0.2,0.3,0.4,0.5]
    n_it = len(num_leaves)*len(max_depth)*len(learning_rate)*len(n_estimators)*len(max_bin)*len(colsample_bytree)

    HPO_Scores = np.zeros((n_it,8))
    current_it=0
    for nl in num_leaves:
        for md in max_depth:
            for lr in learning_rate:
                for ne in n_estimators:
                    for mb in max_bin:
                        for cb in colsample_bytree:
                            RMSE, MAPE = TestModel(lgb.LGBMRegressor(objective='regression',num_leaves=nl, max_depth=md,
                                  learning_rate=lr, n_estimators=ne,
                                  max_bin = mb, colsample_bytree=cb,
                                  feature_fraction_seed=9, verbose=-1), 10, 0.2, False)
                            HPO_Scores[current_it,:] = np.array([nl,md,lr,ne,mb,cb,RMSE,MAPE])
                            current_it += 1
                            rmse_min = min(HPO_Scores[:,6][HPO_Scores[:,6] != 0])
                            print(str(current_it) + "/" + str(n_it) + " | " + str([round(n,2) for n in [nl,md,lr,ne,mb,cb,RMSE,MAPE]])
                                  + " | Best RMSE: " + str(int(rmse_min)) + " => MAPE: " + str(round(HPO_Scores[HPO_Scores[:,6] == rmse_min][:,7][0],3)) )
def TuneXGB():
    # Tune XGB hyperparameters:
    #220/6561 | [0.35, 0.01, 3, 3, 3000, 0.2, 0.85, 0.5, 17906.88, 7.79] | Best RMSE: 17906 => MAPE: 7.789
    #249/4374 | [0.2, 0.01, 3, 2, 2200, 0.1, 0.85, 0.6, 17805.29, 7.36] | Best RMSE: 17805 => MAPE: 7.356
    #97/144 | [0.2, 0.01, 3, 2, 1500, 0.2, 0.85, 0.5, 17878.56, 7.8] | Best RMSE: 17878 => MAPE: 7.797
    #2/12 | [0.2, 0.01, 3, 2, 1000, 0.1, 0.85, 0.5, 20586.5, 8.02] | Best RMSE: 20586 => MAPE: 8.023 #COM 20

    colsample_bytree = [0.4603]
    gamma = [0.0468]
    max_depth = [3]
    min_child_weight = [1.7817]
    n_estimators = [2200]
    reg_alpha = [0.4640]
    reg_lambda = [0.8571]
    subsample = [0.5213]

    n_it = len(colsample_bytree) * len(gamma) * len(max_depth) * len(min_child_weight) * len(n_estimators) * len(
        reg_alpha)*len(reg_lambda)*len(subsample)

    HPO_Scores = np.zeros((n_it, 10))
    current_it = 0
    for cb in colsample_bytree:
        for g in gamma:
            for md in max_depth:
                for mcw in min_child_weight:
                    for ne in n_estimators:
                        for ra in reg_alpha:
                                for rl in reg_lambda:
                                    for s in subsample:
                                        RMSE, MAPE = TestModel(
                                            xgb.XGBRegressor(colsample_bytree=cb, gamma=g,
                                                             learning_rate=0.05, max_depth=md,
                                                             min_child_weight=mcw, n_estimators=ne,
                                                             reg_alpha=ra, reg_lambda=rl,
                                                             subsample=s,
                                                             random_state=7, nthread=-1), 20, 0.2, False)
                                        HPO_Scores[current_it, :] = np.array([cb, g, md, mcw, ne, ra, rl, s, RMSE, MAPE])
                                        current_it += 1
                                        rmse_min = min(HPO_Scores[:, -2][HPO_Scores[:, -2] != 0])
                                        print(str(current_it) + "/" + str(n_it) + " | " + str(
                                            [round(n, 2) for n in [cb, g, md, mcw, ne, ra, rl, s, RMSE, MAPE]])
                                              + " | Best RMSE: " + str(int(rmse_min)) + " => MAPE: " + str(
                                            round(HPO_Scores[HPO_Scores[:, -2] == rmse_min][:, -1][0], 3)))
#test commit 2 2
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

quantitative = ['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'PoolArea',
 'MiscVal',
 'MoSold',
 'YrSold']
qualitative = ['MSZoning',
 'Street',
 'Alley',
 'LotShape',
 'LandContour',
 'Utilities',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'ExterQual',
 'ExterCond',
 'Foundation',
 'BsmtQual',
 'BsmtCond',
 'BsmtExposure',
 'BsmtFinType1',
 'BsmtFinType2',
 'Heating',
 'HeatingQC',
 'CentralAir',
 'Electrical',
 'KitchenQual',
 'Functional',
 'FireplaceQu',
 'GarageType',
 'GarageFinish',
 'GarageQual',
 'GarageCond',
 'PavedDrive',
 'PoolQC',
 'Fence',
 'MiscFeature',
 'SaleType',
 'SaleCondition']
# test_normality = lambda x: shapiro(x.fillna(0))[1] < 0.01
# normal = pd.DataFrame(df_train[quantitative])
# normal = normal.apply(test_normality)
# print(not normal.any())
# df_train[quantitative] = normal

for col in df_train.columns[1:-1]:
    if col in quantitative:
        print("")
        # df_train[col] = df_train[col].fillna(0)
        # df_test[col] = df_test[col].fillna(0)
    else:
        df_train[col] = df_train[col].fillna('None')
        df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))
        df_test[col] = df_test[col].fillna('None')
        df_test[col] = LabelEncoder().fit_transform(df_test[col].astype(str))
#
# df_train = MultiColumnLabelEncoder(columns = ['MSSubClass','Alley','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence','MiscFeature','PoolQC','SaleType','SaleCondition']).fit_transform(df_train)
# df_test = MultiColumnLabelEncoder(columns = ['MSSubClass','Alley','MSZoning','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive','Fence','MiscFeature','PoolQC','SaleType','SaleCondition']).fit_transform(df_test)

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
    ColumnsToCheck = list(df_train.columns)[1:-1]
    # #ColumnsToCheck = ['GrLivArea']
    # for col in ColumnsToCheck:
    #     fig, ax = plt.subplots()
    #     ax.scatter(x = df_train[col], y = df_train['SalePrice'])
    #     plt.ylabel('SalePrice', fontsize=13)
    #     plt.xlabel(col, fontsize=13)
    #     plt.show()
    #     print(str(col) + ' is clean!')
    #     print(" ")
################### Feature ENgineering ###################
if True:
    # df_train_old = df_train.copy()
    #
    # numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    # numerics2 = []
    # for i in df_train.columns[1:-1]:
    #     if df_train[i].dtype in numeric_dtypes:
    #         numerics2.append(i)
    # skew_features_train = df_train[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)
    # high_skew_train = skew_features_train[skew_features_train > 0.5]
    # skew_index_train = high_skew_train.index
    #
    # for i in skew_index_train:
    #     df_train[i] = boxcox1p(df_train[i], boxcox_normmax(df_train[i] + 1))
    #     df_test[i] = boxcox1p(df_test[i], boxcox_normmax(df_test[i] + 1))


    df_train = df_train.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

    df_train['YrBltAndRemod']=df_train['YearBuilt']+df_train['YearRemodAdd']
    df_train['TotalSF']=df_train['TotalBsmtSF'] + df_train['1stFlrSF'] + df_train['2ndFlrSF']

    df_train['Total_sqr_footage'] = (df_train['BsmtFinSF1'] + df_train['BsmtFinSF2'] +
                                     df_train['1stFlrSF'] + df_train['2ndFlrSF'])

    df_train['Total_Bathrooms'] = (df_train['FullBath'] + (0.5 * df_train['HalfBath']) +
                                   df_train['BsmtFullBath'] + (0.5 * df_train['BsmtHalfBath']))

    df_train['Total_porch_sf'] = (df_train['OpenPorchSF'] + df_train['3SsnPorch'] +
                                  df_train['EnclosedPorch'] + df_train['ScreenPorch'] +
                                  df_train['WoodDeckSF'])

    df_train['haspool'] = df_train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df_train['has2ndfloor'] = df_train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df_train['hasgarage'] = df_train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df_train['hasbsmt'] = df_train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df_train['hasfireplace'] = df_train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    df_test = df_test.drop(['Utilities', 'Street', 'PoolQC',], axis=1)

    df_test['YrBltAndRemod']=df_test['YearBuilt']+df_test['YearRemodAdd']
    df_test['TotalSF']=df_test['TotalBsmtSF'] + df_test['1stFlrSF'] + df_test['2ndFlrSF']

    df_test['Total_sqr_footage'] = (df_test['BsmtFinSF1'] + df_test['BsmtFinSF2'] +
                                     df_test['1stFlrSF'] + df_test['2ndFlrSF'])

    df_test['Total_Bathrooms'] = (df_test['FullBath'] + (0.5 * df_test['HalfBath']) +
                                   df_test['BsmtFullBath'] + (0.5 * df_test['BsmtHalfBath']))

    df_test['Total_porch_sf'] = (df_test['OpenPorchSF'] + df_test['3SsnPorch'] +
                                  df_test['EnclosedPorch'] + df_test['ScreenPorch'] +
                                  df_test['WoodDeckSF'])

    df_test['haspool'] = df_test['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df_test['has2ndfloor'] = df_test['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df_test['hasgarage'] = df_test['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df_test['hasbsmt'] = df_test['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df_test['hasfireplace'] = df_test['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

    # print(df_train.shape)
    df_train = pd.get_dummies(df_train).reset_index(drop=True)
    # print(df_train.shape)

    outliers = [30, 88, 462, 631, 1322]
    df_train = df_train.drop(df_train.index[outliers])

    overfit = []
    for i in df_train.drop(columns="SalePrice").iloc[:,1:].columns:
        counts = df_train.drop(columns="SalePrice").iloc[:,1:][i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df_train.drop(columns="SalePrice").iloc[:,1:]) * 100 > 99.94:
            overfit.append(i)

    overfit = list(overfit)
    # overfit.append('MSZoning_C (all)')

    df_train.drop(columns="SalePrice").iloc[:,1:] = df_train.drop(columns="SalePrice").iloc[:,1:].drop(overfit, axis=1).copy()


## Modelling
# kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
#
# def rmsle(y, y_pred):
#     return np.sqrt(mean_squared_error(y, y_pred))
#
# def cv_rmse(model):
#     rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
#     return (rmse)
# alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
# alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
# e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
# e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
#
# ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
# lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
# elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
# svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
# gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)
# lightgbm = lgb.LGBMRegressor(objective='regression',
#                                        num_leaves=4,
#                                        learning_rate=0.01,
#                                        n_estimators=5000,
#                                        max_bin=200,
#                                        bagging_fraction=0.75,
#                                        bagging_freq=5,
#                                        bagging_seed=7,
#                                        feature_fraction=0.2,
#                                        feature_fraction_seed=7,
#                                        verbose=-1,
#                                        )
# xgboost = xgb.XGBRegressor(learning_rate=0.01,n_estimators=3460,
#                                      max_depth=3, min_child_weight=0,
#                                      gamma=0, subsample=0.7,
#                                      colsample_bytree=0.7,
#                                      objective='reg:linear', nthread=-1,
#                                      scale_pos_weight=1, seed=27,
#                                      reg_alpha=0.00006)
# stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
#                                 meta_regressor=xgboost,
#                                 use_features_in_secondary=True)

# score = cv_rmse(ridge)
# print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
# imp_median = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

lasso = make_pipeline(RobustScaler(), imp_median, Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), imp_median, ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = make_pipeline(RobustScaler(), imp_median, KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
svr = make_pipeline(RobustScaler(), imp_median, SVR(C= 20, epsilon= 0.008, gamma=0.0003,))

GBoost = make_pipeline(imp_median, GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10,
                                   loss='huber', random_state =5))
# Tuned XGB Model [0.2, 0.01, 3, 2, 1000, 0.1, 0.85, 0.5, 20586.5, 8.02]
model_xgb = xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.01,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=2, n_estimators=1000,
                             reg_alpha=0.1, reg_lambda=0.85,
                             subsample=0.5, random_state =7, nthread = -1)
# Tuned LGB Model
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5, max_depth=-1,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 40, colsample_bytree=0.3,
                              feature_fraction_seed=9, verbose=-1)

# Stack model
baggingmodel_lasso = BaggingRegressor(base_estimator=lasso)
baggingmodel_ENet = BaggingRegressor(base_estimator=ENet)
# baggingmodel_KRR = BaggingRegressor(base_estimator=KRR)
baggingmodel_svr = BaggingRegressor(base_estimator=svr)
baggingmodel_GBoost = BaggingRegressor(base_estimator=GBoost)

GBoost = GBoost
baggingmodel_lgb = BaggingRegressor(base_estimator=model_lgb)
baggingmodel_xgb = BaggingRegressor(base_estimator=model_xgb)
stackmodel = make_pipeline(imp_median,StackingCVRegressor(regressors=(lasso, ENet, GBoost,
                                             model_xgb, model_lgb),
                                meta_regressor=model_xgb,
                                use_features_in_secondary=True))

# TestModel(stackmodel, 1, 0.20,True)
#stackmodel: RMSLE: 0.012 | MAPE: 7.785
#baggingmodel_xgb: RMSLE: 0.012 | MAPE: 7.482
#baggingmodel_lgb:RMSLE: 0.012 | MAPE: 7.568
#baggingmodel_GBoost:
#GBoost:

# sbaggingmodel_lasso = MakePrediction(baggingmodel_lasso);print("sbaggingmodel_lasso done")
# sbaggingmodel_ENet = MakePrediction(baggingmodel_ENet);print("baggingmodel_ENet done")
# sbaggingmodel_KRR = MakePrediction(baggingmodel_KRR);print("baggingmodel_KRR done")
# sbaggingmodel_svr = MakePrediction(baggingmodel_svr);print("baggingmodel_svr done")
sGBoost = MakePrediction(baggingmodel_GBoost);print("baggingmodel_GBoost done")
sbaggingmodel_lgb = MakePrediction(baggingmodel_lgb);print("baggingmodel_lgb done")
sbaggingmodel_xgb = MakePrediction(baggingmodel_xgb);print("baggingmodel_xgb done")
sstack = MakePrediction(stackmodel);print("stackmodel done")

# sbaggingmodel_lasso.to_csv("submission5.csv", index=False)

#Final Ensemble Prediction
# final_submission = pd.concat([0.1*sbaggingmodel_lasso["SalePrice"],0.5*sbaggingmodel_ENet["SalePrice"],
#                               0.3*sGBoost["SalePrice"],0.1*sbaggingmodel_xgb["SalePrice"],0.1*sbaggingmodel_lgb["SalePrice"],
#                               0.10*sstack["SalePrice"]], axis=1).sum(axis=1)
final_submission = pd.concat(
    [0.25*sGBoost["SalePrice"],0.25*sbaggingmodel_xgb["SalePrice"],0.25*sbaggingmodel_lgb["SalePrice"],0.25*sstack["SalePrice"]]
    , axis=1).sum(axis=1)
final_submission = pd.concat(sbaggingmodel_lgb["SalePrice"], axis=1).sum(axis=1)
s = sbaggingmodel_xgb.copy()
s["SalePrice"] = final_submission
submission = s
submission.to_csv("submission6.csv", index=False)


print("END")