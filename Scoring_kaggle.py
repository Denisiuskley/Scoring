try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from catboost import CatBoostClassifier, Pool
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from pandas_profiling import ProfileReport


def percentil5 (x):
    x = x.dropna()
    if x.shape[0] > 0:
        return np.round(np.percentile(x, q=5), 3)
    else: return 0
def percentil95 (x): 
    x = x.dropna()
    if x.shape[0] > 0:
        return np.round(np.percentile(x, q=95), 3)
    else: return 0

def group_stat(df, group, for_stat):
    gr = df.groupby(group).agg(
        Par_min = (for_stat, 'min'),
        Par_quantil1 = (for_stat, percentil5),
        Par_median = (for_stat, 'median'),
        Par_mean = (for_stat, 'mean'),
        Par_quantil3 = (for_stat, percentil95),
        Par_max = (for_stat, 'max'),
        Par_sum = (for_stat, 'sum'),
        Par_count = (for_stat, 'count')).reset_index()
    return gr

df = pd.read_csv('train.csv')
dft = pd.read_csv('test.csv')
dft['default'] = np.nan
df = df.append(dft)

#profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
#profile.to_file(output_file="Report.html")
'''В отчете можно посмотреть подробное описание переменных, наличие пропусков и выбросов'''

cleanup_nums = {'education': {'ACD': 0, 'PGR': 1, 'UGR': 2, 'GRD': 3, 'SCH': 4},
                'sex': {'M': 0, 'F': 1},
                'car': {'N': 0, 'Y': 1},
                'car_type': {'N': 0, 'Y': 1},
                'foreign_passport': {'N': 0, 'Y': 1}}

#cleanup_nums = {'sex': {'M': 0, 'F': 1},
#                'car': {'N': 0, 'Y': 1},
#                'car_type': {'N': 0, 'Y': 1},
#                'foreign_passport': {'N': 0, 'Y': 1}}

df.replace(cleanup_nums, inplace=True)

#Преобразование переменной для учета отечественных авто
df['car_type'] = df.apply(lambda x: 2 if x.car == 1 and x.car_type == 0 else x.car_type, axis = 1)

'''************Заполнение пропусков******************'''
dft = df.dropna(subset = ['education'])

X = dft.drop(['client_id', 'education', 'default', 'app_date'], axis = 1)
y = dft['education']

cols = list(X.columns)
pred = df[df['education'].isnull()]
X_pred= pred.drop(['client_id', 'education', 'default', 'app_date'], axis = 1)

model = CatBoostClassifier(iterations = 1500,learning_rate = 0.05, depth = 4,
                           l2_leaf_reg = 1, 
                           eval_metric='Accuracy', verbose = 500)
model.fit(X, y, cat_features = [3, 5, 8, 9, 10, 12, 13, 14])
y_pred = model.predict(X_pred)
df.loc[df['education'].isnull(), 'education'] = y_pred
'''***************************************************'''

'''******************Преобразование даты*******************'''
s = pd.to_datetime(df['app_date'])
df['month'] = s.dt.month
vc = df['app_date'].value_counts().reset_index()
vc.columns = ['app_date', 'counts']
df = pd.merge(df, vc, how = 'left', on = 'app_date')
'''***************************************************'''

'''******************Логарифмирование параметров со смещенными распределениями*******************'''
num_log = ['decline_app_cnt', 'bki_request_cnt', 'income']
#num_log = ['income']
for i in num_log:
    df[i] = np.log(df[i] + 1)
'''***************************************************'''

num_cols = ['age', 'decline_app_cnt', 'bki_request_cnt', 'income', 'score_bki']
bin_cols = ['sex', 'car_type', 'good_work', 'foreign_passport']
cat_cols = ['education', 'work_address', 'home_address', 'region_rating', 'sna', 'first_time']

'''******************Проверка значимости*******************'''
for i in num_cols:
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(x = 'default', y = i, data = df)
    ax.set_title(i)

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr().abs(), vmin=0, vmax=1)

df['education'] = df['education'].astype(int)
dft = df[~df['default'].isnull()]
imp_num = pd.Series(f_classif(dft[num_cols], dft['default'])[0], index = num_cols)
imp_num.sort_values(inplace = True)
fig, ax = plt.subplots(figsize=(10, 10))
imp_num.plot(kind = 'barh')

imp_cat = pd.Series(mutual_info_classif(dft[bin_cols + cat_cols], dft['default'],
                                     discrete_features =True), index = bin_cols + cat_cols)
imp_cat.sort_values(inplace = True)
fig, ax = plt.subplots(figsize=(10, 10))
imp_cat.plot(kind = 'barh')
'''***************************************************'''

X = dft.drop(['default', 'client_id', 'app_date', 'car'], axis = 1)
y = dft['default']

'''******************Оптимизация гиперпараметров*******************'''
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#
#def black_box_function(itr, lr, dep, l2):
#    model = CatBoostClassifier(iterations = int(itr),learning_rate = lr, depth = int(dep),
#                                              l2_leaf_reg = int(l2), custom_metric = 'AUC', eval_metric = 'AUC', verbose = False)
#    model.fit(X_train, y_train, cat_features = [0, 1, 3, 5, 8, 9, 10, 12, 13, 14, 15, 16])
#    y_pred = model.predict_proba(X_test)[:,1]
#    return roc_auc_score(y_test, y_pred)
#
#from bayes_opt import BayesianOptimization
#pbounds = {'itr': (1000, 5000), 'lr': (0.01, 0.1), 'dep': (2, 10), 'l2': (1, 10)}
#optimizer = BayesianOptimization(
#    f=black_box_function,
#    pbounds=pbounds,
#    random_state=1,
#)
#optimizer.maximize(
#    init_points=20,
#    n_iter=2,
#)
'''***************************************************'''

'''******************Предварительная оценка качества на 4фолд валидации*******************'''
cols = list(X.columns)
kf = KFold(n_splits=4, shuffle = True, random_state = 0)
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns
i2 = 0
results = []
y_pred_m = np.zeros((y.shape[0]))
for train, test in kf.split(X, y):
    model = CatBoostClassifier(iterations = 1500,learning_rate = 0.01, depth = 4,
                                              l2_leaf_reg = 1, class_weights = [1, 64427/9372], 
                                              custom_metric = 'AUC', eval_metric = 'AUC', verbose = 1000)
    model.fit(X.iloc[train,:], y.iloc[train], cat_features = [0, 1, 3, 5, 8, 9, 10, 12, 13, 14, 15])
#    model.fit(X.iloc[train,:], y.iloc[train], cat_features = [0, 8, 9, 10, 12, 13, 15, 16])
    y_pred = model.predict_proba(X.iloc[test,:])[:,1]
    y_pred_m[test] = y_pred
    feature_importances[i2] = model.get_feature_importance(data=None,
       prettified=False, thread_count=-1, verbose=False)
    roc_auc = roc_auc_score(y[test], y_pred)
    print('ROC_AUC:', np.round(roc_auc, 3))
    results.append(roc_auc)
    i2 += 1
print('CB:', np.round(np.mean(results), 5))
'''***************************************************'''

'''******************Оценка значимости переменных*******************'''
fi_plot = pd.DataFrame()
for i in range(i2):
    temp = feature_importances[['feature',i]]
    temp.columns = ['feature', 'importance']    
    fi_plot = fi_plot.append(temp, sort=False)
feature_importances['mean'] = feature_importances.iloc[:,1:].mean(axis = 1)
fi_plot = pd.merge(fi_plot, feature_importances[['feature', 'mean']], 
                            on=['feature'], how='left')

ind = np.unravel_index(np.argsort(feature_importances['mean'], axis=None), feature_importances['mean'].shape)[0]

plt.figure(figsize=(16, 16))
sns.barplot(data=fi_plot.sort_values(by='mean', ascending=False), x='importance', y='feature', capsize=.2)
'''***************************************************'''
'''Параметрическое обеспечение далеко не полное. Сам по себе доход мало о чем говорит
в отрыве от суммы кредита, его периода и процентной ставки. Также, важно понимать каким
образом клиент считался дефолтным, по каким показателям. Непонятны и некоторые параметры (категориальные),
принцип их расчета.'''

'''Балансировка классов немного улучила оценку на каггле, хотя сильно меняет распределение.
Что выгоднее в реальных условиях, нужно считать экономически.'''
model = CatBoostClassifier(iterations = 1500,learning_rate = 0.01, depth = 4,
                                              l2_leaf_reg = 1, class_weights = [1, 64427/9372],
                                              custom_metric = 'AUC', eval_metric = 'AUC', verbose = 1000)
model.fit(X, y, cat_features = [0, 1, 3, 5, 8, 9, 10, 12, 13, 14, 15])

df_sub = df[df['default'].isnull()]
X_sub = df_sub.drop(['default', 'client_id', 'app_date', 'car'], axis = 1)
y_pred = model.predict_proba(X_sub)[:,1]

output = pd.DataFrame({'client_id': df_sub['client_id'], 'default': y_pred})
output.to_csv('Balanced.csv', index=False)

