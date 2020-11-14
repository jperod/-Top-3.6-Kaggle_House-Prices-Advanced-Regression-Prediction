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

X_train = df_train.drop(['SalePrice','Id'], axis=1)
X_test = df_test.drop(['Id'], axis=1)
X = pd.concat([X_train, X_test]).reset_index(drop=True)

print(END)
