import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, norm
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

from catboost import Pool
from catboost import CatBoostRegressor
import missingno as msno


def impute_knn(df):
    ttn = train_test.select_dtypes(include=[np.number])
    ttc = train_test.select_dtypes(exclude=[np.number])

    cols_nan = ttn.columns[ttn.isna().any()].tolist()
    cols_no_nan = ttn.columns.difference(cols_nan).values

    for col in cols_nan:
        imp_test = ttn[ttn[col].isna()]
        imp_train = ttn.dropna()
        model = KNeighborsRegressor(n_neighbors=5)
        knr = model.fit(imp_train[cols_no_nan], imp_train[col])
        ttn.loc[ttn[col].isna(), col] = knr.predict(imp_test[cols_no_nan])

    return pd.concat([ttn, ttc], axis=1)


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


train = pd.read_csv('AmesHousing.csv')
test = pd.read_csv('test.csv')
train.columns = train.columns.str.replace(' ', '')
train = train.drop(['Order', 'PID'], axis=1)
train.describe()

test.drop('Id', axis=1, inplace=True)

train_test = pd.concat([train, test], axis=0, ignore_index=True)

msno.matrix(train_test)
plt.show()

numerical_features = train_test.select_dtypes(include=[int, float]).columns.tolist()
categorical_features = train_test.select_dtypes(include='object').columns.tolist()

sns.displot(train['SalePrice'], kde=True)
plt.show()

# corr = train_test.corr()
# sns.heatmap(corr, cmap='viridis')

train_test[numerical_features].hist(bins=25, figsize=(15, 15))
plt.show()

selected_features = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(data=train_test[selected_features], size=2)
plt.show()

sns.regplot(data=train_test, x='GrLivArea', y='SalePrice')
plt.show()

sns.regplot(data=train_test, x='TotalBsmtSF', y='SalePrice')
plt.show()

sns.boxplot(data=train_test, y='SalePrice', x='OverallQual')
plt.show()

sns.boxplot(data=train_test, x='YearBuilt', y='SalePrice')
plt.show()

train_test['MSSubClass'] = train_test['MSSubClass'].apply(str)
train['YrSold'] = train_test['YrSold'].apply(str)
train['MoSold'] = train_test['MoSold'].apply(str)

train_test['Functional'] = train_test['Functional'].fillna('Typ')
train_test['Electrical'] = train_test['Electrical'].fillna('SBrkr')
train_test['KitchenQual'] = train_test['KitchenQual'].fillna('TA')
train_test['Exterior1st'] = train_test['Exterior1st'].fillna(train_test['Exterior1st'].mode()[0])
train_test['Exterior2nd'] = train_test['Exterior2nd'].fillna(train_test['Exterior2nd'].mode()[0])
train_test['SaleType'] = train_test['SaleType'].fillna(train_test['SaleType'].mode()[0])
train_test['PoolQC'] = train_test['PoolQC'].fillna('None')
train_test['Alley'] = train_test['Alley'].fillna('None')
train_test['FireplaceQu'] = train_test['FireplaceQu'].fillna('None')
train_test['Fence'] = train_test['Fence'].fillna('None')
train_test['MiscFeature'] = train_test['MiscFeature'].fillna('None')

for col in ('GarageArea', 'GarageCars'):
    train_test[col] = train_test[col].fillna(0)

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train_test[col] = train_test[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train_test[col] = train_test[col].fillna('None')

useless = ['GarageYrBlt', 'YearRemodAdd']
train_test.drop(useless, axis=1, inplace=True)

missing_values = train_test.isnull().sum()
columns_with_missing = missing_values[missing_values > 0].index.tolist()
columns_with_missing = [col for col in columns_with_missing if train_test[col].isnull().any()]
print(f'Columns with missing values: {columns_with_missing}')

train_test = impute_knn(train_test)

objects = []
for i in train_test.columns:
    if train_test[i].dtype == object:
        objects.append(i)
train_test.update(train_test[objects].fillna('None'))

print(train_test[columns_with_missing].isna().sum())

train_test['SqFtPerRoom'] = train_test['GrLivArea'] / (
        train_test['TotRmsAbvGrd'] + train_test['FullBath'] + train_test['HalfBath'] + train_test['KitchenAbvGr'])
train_test['Total Home Quality'] = train_test['OverallQual'] + train_test['OverallCond']
train_test['Total Bathrooms'] = train_test['FullBath'] + (0.5 * train_test['HalfBath']) + train_test['BsmtFullBath'] + (
        0.5 * train_test['BsmtHalfBath'])

train_test['HighQualSF'] = train_test['GrLivArea'] + train_test['1stFlrSF'] + train_test['2ndFlrSF'] + 0.5 * train_test[
    'GarageArea'] + 0.5 * train_test['TotalBsmtSF'] + train_test['MasVnrArea']

train_test['Age'] = pd.to_numeric(train_test['YrSold']) - pd.to_numeric(train_test['YearBuilt'])
train_test['Renovate'] = pd.to_numeric(train_test['YearRemod/Add']) - pd.to_numeric(train_test['YearBuilt'])

train_test_dummy = pd.get_dummies(train_test)
print(train_test_dummy)

numeric_features = train_test_dummy.dtypes[train_test_dummy.dtypes != object].index
skewed_features = train_test_dummy[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_features[skewed_features > 0.5]
skew_index = high_skew.index

for i in skew_index:
    train_test_dummy[i] = np.log1p(train_test_dummy[i])

target = train['SalePrice']
target_log = np.log1p(target)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('QQ plot and distribution Sale Price', fontsize=15)
sm.qqplot(target_log, stats.t, distargs=(4,), fit=True, line='45', ax=ax[0])
sns.distplot(target_log, kde=True, hist=True, fit=norm, ax=ax[1])
plt.show()

HighQualSF_log = np.log1p(train_test['HighQualSF'])
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('QQ plot and the distribution Sale Price', fontsize=15)
sm.qqplot(HighQualSF_log, stats.t, distargs=(4,), fit=True, line='45', ax=ax[0])
sns.distplot(HighQualSF_log, kde=True, hist=True, fit=norm, ax=ax[1])
plt.show()

train_test['HighQualSF'] = HighQualSF_log

GrLivArea_log = np.log1p(train_test['GrLivArea'])
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('QQ plot and distribution Sale Price', fontsize=15)

sm.qqplot(GrLivArea_log, stats.t, distargs=(4,), fit=True, line='45', ax=ax[0])
sns.distplot(GrLivArea_log, kde=True, hist=True, fit=norm, ax=ax[1])
plt.show()

train_test['GrLivArea'] = GrLivArea_log

train = train_test_dummy[0:2930]
test = train_test_dummy[2930:]
test.drop('SalePrice', axis=1, inplace=True)

ytrain = target_log
xtrain = train.drop('SalePrice', axis=1)

X_train, X_val, y_train, y_val = train_test_split(xtrain, ytrain, test_size=0.5, random_state=42)
X_train, y_train = xtrain, ytrain

print(X_train)
print(X_val)

final_model = CatBoostRegressor(random_seed=42)
final_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
print(final_model.get_all_params())

final_pred = final_model.predict(X_val)
final_score = rmse(y_val, final_pred)
print(final_score)

submission = pd.read_csv('submission.csv')
test_pred = np.expm1(final_model.predict(test))
submission['SalePrice'] = test_pred
print(submission)

submission.to_csv('submission.csv', index=False, header=True)
