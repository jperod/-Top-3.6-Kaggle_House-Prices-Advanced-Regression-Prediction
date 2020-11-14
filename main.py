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
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
y = df_train['SalePrice']

ID_train = df_train["Id"]
ID_test = df_test["Id"]
X_train = df_train.drop(['SalePrice','Id'], axis=1)
X_test = df_test.drop(['Id'], axis=1)
X = pd.concat([X_train, X_test]).reset_index(drop=True)

y = np.log1p(y)

#Check for outliers
from sklearn.ensemble import IsolationForest
outliers = IsolationForest(random_state=0, contamination="auto").fit_predict(pd.get_dummies(X_train).fillna(0))
outliers = np.squeeze(np.argwhere(outliers==-1))
# print(outliers)

X = X.drop(X.index[outliers])
X_train = X_train.drop(X_train.index[outliers])
y = y.drop(y.index[outliers])
ID_train = ID_train.drop(y.index[outliers])

#Visualize all features and analyze for Data Type & Missing Values

X['MSSubClass'] = X['MSSubClass'].apply(str).fillna(X['MSSubClass'].mode()[0])
X['MSZoning'] = X['MSZoning'].apply(str).fillna(X['MSZoning'].mode()[0])
X['LotFrontage'] = X['LotFrontage'].apply(float).fillna(X['LotFrontage'].median())
X['LotArea'] = X['LotArea'].apply(float).fillna(X['LotArea'].median())
X['Street'] = X['Street'].apply(str).fillna(X['Street'].mode()[0])
X['Alley'] = X['Alley'].apply(str)
X['Utilities'] = X['Utilities'].apply(str)
X['Exterior1st'] = X['Exterior1st'].apply(str)
X['Exterior2nd'] = X['Exterior2nd'].apply(str)
X['MasVnrType'] = X['MasVnrType'].apply(str)
X['MasVnrArea'] = X['MasVnrArea'].apply(float).fillna(X['LotFrontage'].median())
X['BsmtQual'] = X['BsmtQual'].apply(str)
X['BsmtCond'] = X['BsmtCond'].apply(str)
X['BsmtExposure'] = X['BsmtExposure'].apply(str)
X['BsmtFinType1'] = X['BsmtFinType1'].apply(str)
X['BsmtFinSF1'] = X['BsmtFinSF1'].apply(float).fillna(X['BsmtFinSF1'].median())
X['BsmtFinType2'] = X['BsmtFinType2'].apply(str)
X['BsmtFinSF2'] = X['BsmtFinSF2'].apply(float).fillna(X['BsmtFinSF2'].median())
X['BsmtUnfSF'] = X['BsmtUnfSF'].apply(float).fillna(X['BsmtFinSF2'].median())
X['TotalBsmtSF'] = X['TotalBsmtSF'].apply(float).fillna(X['TotalBsmtSF'].median())
X['Electrical'] = X['Electrical'].apply(str)
X['BsmtFullBath'] = X['BsmtFullBath'].apply(str)
X['BsmtHalfBath'] = X['BsmtHalfBath'].apply(str)
X['KitchenQual'] = X['KitchenQual'].apply(str)
X['Functional'] = X['Functional'].apply(str)
X['FireplaceQu'] = X['FireplaceQu'].apply(str)
X['GarageType'] = X['GarageType'].apply(str)
X['GarageYrBlt'] = X['GarageYrBlt'].apply(float).fillna(0)
X['GarageFinish'] = X['GarageFinish'].apply(str)
X['GarageCars'] = X['GarageCars'].apply(str)
X['GarageArea'] = X['GarageArea'].apply(float).fillna(0)
X['GarageQual'] = X['GarageQual'].apply(str)
X['GarageCond'] = X['GarageCond'].apply(str)
X['PoolQC'] = X['PoolQC'].apply(str)
X['Fence'] = X['Fence'].apply(str)
X['MiscFeature'] = X['MiscFeature'].apply(str)
X['SaleType'] = X['SaleType'].apply(str)

X_train = X.iloc[0:X_train.shape[0],:]
c=-1
overfit = []
for col in X.columns:
    c+=1
    if c>=0:
        # print(str(col) + " " + str(c))
        fig, ax = plt.subplots()
        ax.scatter(x = X_train[col], y = y)
        plt.ylabel('SalePrice', fontsize=13)
        plt.xlabel(col, fontsize=13)
        plt.title(col)
        # plt.show()
        missingvalues = X[col].isnull().values.any()
        counts = X[col].value_counts()
        BMAX = max(counts) / len(X) * 100
        if BMAX == 100:
            overfit.append(col)
        # print(" MV: " + str(missingvalues) + " | " + str(BMAX))
        # print("")

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in X.columns:
    if X[i].dtype in numeric_dtypes:
        numerics2.append(i)
skew_X = X[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_X[skew_X > 0.5]
skew_index = high_skew.index

for i in skew_index:
    X[i] = boxcox1p(X[i], boxcox_normmax(X[i] + 1))

X['YrBltAndRemod']=X['YearBuilt']+X['YearRemodAdd']
X['TotalSF']=X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

X['Total_sqr_footage'] = (X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF'])

X['Total_Bathrooms'] = (X['FullBath'] + (0.5 * X['HalfBath'].astype(int)) +X['BsmtFullBath'].astype(float) + (0.5 * X['BsmtHalfBath'].astype(float)))

X['Total_porch_sf'] = (X['OpenPorchSF'] + X['3SsnPorch'] +X['EnclosedPorch'] + X['ScreenPorch'] +X['WoodDeckSF'])

X['haspool'] = X['PoolArea'].apply([lambda x: 1 if x > 0 else 0])
X['has2ndfloor'] = X['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
X['hasgarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
X['hasbsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
X['hasfireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

X = pd.get_dummies(X).reset_index(drop=True)

X_train = X.iloc[0:X_train.shape[0],:]
X_test = X.iloc[-X_test.shape[0]:,:]
kfolds = KFold(n_splits=10, shuffle=True, random_state=0)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from sklearn.metrics import make_scorer
mape_error = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, np.array(X_train), np.array(y), scoring="neg_mean_squared_error", cv=kfolds))
    mape = -cross_val_score(model, np.array(X_train), np.array(y), scoring=mape_error, cv=kfolds)
    return (rmse, mape)

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

from sklearn.impute import SimpleImputer
imp_median = SimpleImputer(missing_values=np.nan, strategy="constant", fill_value=0)

ridge = make_pipeline(RobustScaler(), imp_median, RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), imp_median, LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), imp_median,  ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))
svr = make_pipeline(RobustScaler(), imp_median, SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
gbr = make_pipeline(imp_median, GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42))
lightgbm = BaggingRegressor(base_estimator=lgb.LGBMRegressor(objective='regression',num_leaves=5, max_depth=-1,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 40, colsample_bytree=0.3,
                              feature_fraction_seed=9, verbose=-1))
xgboost = BaggingRegressor(base_estimator=xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.01,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=2, n_estimators=1000,
                             reg_alpha=0.1, reg_lambda=0.85,
                             subsample=0.5, random_state =7, nthread = -1))
stack_gen = make_pipeline(imp_median, StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgb.XGBRegressor(colsample_bytree=0.2, gamma=0.01,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=2, n_estimators=1000,
                             reg_alpha=0.1, reg_lambda=0.85,
                             subsample=0.5, random_state =7, nthread = -1), lgb.LGBMRegressor(objective='regression',num_leaves=5, max_depth=-1,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 40, colsample_bytree=0.3,
                              feature_fraction_seed=9, verbose=-1)),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True))
score, score2 = cv_rmse(ridge)
print("RIDGE: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

score, score2 = cv_rmse(lasso)
print("LASSO: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

score, score2 = cv_rmse(elasticnet)
print("ENet: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

score, score2 = cv_rmse(svr)
print("SVR: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

score, score2 = cv_rmse(lightgbm)
print("LGB: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

score, score2 = cv_rmse(gbr)
print("GBR: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

score, score2 = cv_rmse(xgboost)
print("XGB: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

score, score2 = cv_rmse(stack_gen)
print("STACK_MODEL: RMSE {:.4f} | MAPE {:.4f})\n".format(score.mean(), score2.mean()))

def MakePrediction(model, X, y, X_test):
    model.fit(X, y)
    y_pred = np.expm1(model.predict(np.array(X_test)))
    return pd.DataFrame(y_pred, columns=['SalePrice'])

print('elasticnet')
Pred_Elastic = MakePrediction(elasticnet, np.array(X_train), np.array(y), X_test)
print('Lasso')
Pred_Lasso = MakePrediction(lasso, np.array(X_train), np.array(y), X_test)
print('Ridge')
Pred_Ridge = MakePrediction(ridge, np.array(X_train), np.array(y), X_test)
print('Svr')
Pred_Svr = MakePrediction(svr, np.array(X_train), np.array(y), X_test)
print('GradientBoosting')
Pred_Gbr = MakePrediction(gbr, np.array(X_train), np.array(y), X_test)
print('xgboost')
Pred_Xgb = MakePrediction(xgboost, np.array(X_train), np.array(y), X_test)
print('lightgbm')
Pred_lgb = MakePrediction(lightgbm, np.array(X_train), np.array(y), X_test)
print('stack_gen')
PredStack = MakePrediction(stack_gen, np.array(X_train), np.array(y), X_test)
#
final_pred = pd.concat([0.15*Pred_Elastic["SalePrice"],
                        0.15*Pred_Lasso["SalePrice"],
                        0.15*Pred_Ridge["SalePrice"],
                        0.10*Pred_Svr["SalePrice"],
                        0.05*Pred_Gbr["SalePrice"],
                        0.15*Pred_Xgb["SalePrice"],
                        0.10*Pred_lgb["SalePrice"],
                        0.15*PredStack["SalePrice"]], axis=1).sum(axis=1)
submission = pd.read_csv("sample_submission.csv")
submission["SalePrice"] = final_pred
submission.to_csv("submission7.csv", index=False)
print("END")
