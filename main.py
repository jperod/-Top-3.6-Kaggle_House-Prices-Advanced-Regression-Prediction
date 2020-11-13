import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy.stats import skew #for some statistics
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
from sklearn.model_selection import train_test_split
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
        X_train, X_val, Y_train, Y_val = train_test_split(df_train.iloc[:, 1:-1], df_train.iloc[:, -1], test_size=val_size, shuffle=True)
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
            if False:
                numeric_feats_train = X_train.dtypes[X_train.dtypes != "object"].index
                numeric_feats_val = X_val.dtypes[X_val.dtypes != "object"].index

                # Check the skew of all numerical features
                skewed_feats_train = X_train[numeric_feats_train].apply(lambda x: skew(x.dropna())).sort_values(
                    ascending=False)
                skewed_feats_val = X_val[numeric_feats_val].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
                # print("\nSkew in numerical features: \n")
                skewness_train = pd.DataFrame({'Skew': skewed_feats_train})
                skewness_train = skewness_train[abs(skewness_train) > 0.75]
                skewness_val = pd.DataFrame({'Skew': skewed_feats_val})
                skewness_val = skewness_val[abs(skewness_val) > 0.75]

                skewed_features_train = skewness_train.index
                skewed_features_val = skewness_val.index

                lam = 0.15
                for feat in skewed_features_train:
                    # all_data[feat] += 1
                    X_train[feat] = boxcox1p(X_train[feat], lam)
                for feat in skewed_features_val:
                    # all_data[feat] += 1
                    X_val[feat] = boxcox1p(X_val[feat], lam)
            # Remove NAN
            if False:
                X_train = X_train.fillna(-999)
                X_val = X_val.fillna(-999)

        model.fit(X_train, Y_train)
        y_pred = np.expm1(model.predict(X_val))
        rmse = mean_squared_log_error(Y_val, y_pred)
        mape = mean_absolute_percentage_error(Y_val, y_pred)
        RMSE_list.append(rmse)
        MAPE_list.append(mape)

    if verbose:
        print('RMSE: ' + str(round(np.mean(RMSE_list),3)) + ' | MAPE: ' + str(round(np.mean(MAPE_list),3)))
    return round(np.mean(RMSE_list),3), round(np.mean(MAPE_list),3)
def MakePrediction(model):
    if False:
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
        if False:
            numeric_feats_train = X_train.dtypes[X_train.dtypes != "object"].index
            numeric_feats_val = X_val.dtypes[X_val.dtypes != "object"].index

            # Check the skew of all numerical features
            skewed_feats_train = X_train[numeric_feats_train].apply(lambda x: skew(x.dropna())).sort_values(
                ascending=False)
            skewed_feats_val = X_val[numeric_feats_val].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
            # print("\nSkew in numerical features: \n")
            skewness_train = pd.DataFrame({'Skew': skewed_feats_train})
            skewness_train = skewness_train[abs(skewness_train) > 0.75]
            skewness_val = pd.DataFrame({'Skew': skewed_feats_val})
            skewness_val = skewness_val[abs(skewness_val) > 0.75]

            skewed_features_train = skewness_train.index
            skewed_features_val = skewness_val.index

            lam = 0.15
            for feat in skewed_features_train:
                # all_data[feat] += 1
                X_train[feat] = boxcox1p(X_train[feat], lam)
            for feat in skewed_features_val:
                # all_data[feat] += 1
                X_val[feat] = boxcox1p(X_val[feat], lam)
        # Remove NAN
        if False:
            X_train = X_train.fillna(-999)
            X_val = X_val.fillna(-999)

    X_train, Y_train = df_train.iloc[:,1:-1],df_train.iloc[:,-1]
    Y_train = np.log1p(Y_train)
    X_test = df_test.iloc[:,1:]
    model.fit(X_train, Y_train)
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

# estimators = [('GBoost',GBoost),('lasso',lasso),('ENet',ENet),('KRR',KRR),('xgb', model_xgb),('lgb', model_lgb)]
estimators = [('GBoost',GBoost),('lasso',lasso),('ENet',ENet),('KRR',KRR),('xgb', model_xgb),('lgb', model_lgb)]

stackmodel_1 = StackingRegressor(estimators=estimators,final_estimator=model_xgb)

# TestModel(model_lgb, 20, 0.20,True)
submission = MakePrediction(model_lgb)
submission.to_csv("submission1.csv", index=False)

#1 estimators = [('GBoost',GBoost),('lasso',lasso),('ENet',ENet),('KRR',KRR),('xgb', model_xgb),('lgb', model_lgb)]
# stackmodel_1 = StackingRegressor(estimators=estimators,final_estimator=model_lgb)
# RMSE: 21651.245 | MAPE: 8.206
#1 estimators = [('GBoost',GBoost),('lasso',lasso),('ENet',ENet),('KRR',KRR),('xgb', model_xgb),('lgb', model_lgb)]
# stackmodel_1 = StackingRegressor(estimators=estimators,final_estimator=KRR)
# RMSE: 24939.494 | MAPE: 8.091
#1 estimators = [('GBoost',GBoost),('lasso',lasso),('ENet',ENet),('KRR',KRR),('xgb', model_xgb),('lgb', model_lgb)]
# stackmodel_1 = StackingRegressor(estimators=estimators,final_estimator=model_xgb)
#
print("END")