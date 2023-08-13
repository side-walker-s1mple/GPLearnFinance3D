### 导入相关内容
import numpy as np
import pandas as pd
import genetic
from IPython.core.interactiveshell import InteractiveShell
import warnings
from scipy import stats

np.random.seed(10)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
warnings.filterwarnings('ignore')
InteractiveShell.ast_node_interactivity = "all"
import random

from add_ts_function import dynamic_ts_std, dynamic_ts_mean, dynamic_ts_max
from functions import _function_map
from add_ts_function import _extra_function_map
import os
import sys
sys.path.append(r'..')
from agency_portfolio.Portfolio_track import Portfolio_track
#%% func
def make_XY(df, index_name, columns_name, Y_column1,):
    '''
    return: X: ndarray[n_dates, n_feature, n_stocks], Y: ndarray[n_dates, n_stocks], X_feature_names
    '''
    df = df.pivot_table(index=[index_name], columns=[columns_name], sort=True, dropna=False)
    Y1 = df.loc[:,(Y_column1,)].to_numpy(dtype=np.double)

    df = df.drop([Y_column1,],axis=1)
    X_0_len = len(df.index)
    # df.columns.levels[0] not change after drop
    X_1_len = len(df.columns.levels[0]) - 1
    X_2_len = len(df.columns.levels[1])
    return df.to_numpy(dtype=np.double).reshape((X_0_len, X_1_len, X_2_len)), Y1, df.columns.levels[0].drop([Y_column1,])

numbers = 0
total_df = pd.DataFrame()
input_features = ["股票代码","交易日期","开盘价","最高价","最低价","收盘价","成交量","成交额","收益率"]
different_axis = ("交易日期","股票代码", "收益率",)
################################################################################################
# 构建训练集
total_train_df = pd.DataFrame()
begin_year = pd.Timestamp('2010-01-01')
stop_year = pd.Timestamp('2020-01-01')

# 构建验证集
total_eval_df = pd.DataFrame()
eval_begin_year = pd.Timestamp('2020-01-01')
eval_stop_year = pd.Timestamp('2023-01-01')

# 加载所有股票
stock_data = Portfolio_track()
_close = stock_data.factor_clean(getattr(stock_data, '不复权收盘价'))
_open = stock_data.factor_clean(getattr(stock_data, '不复权开盘价'))
_high = stock_data.factor_clean(getattr(stock_data, '不复权最高价'))
_low = stock_data.factor_clean(getattr(stock_data, '不复权最低价'))
_close_dividend = stock_data.factor_clean(getattr(stock_data, '复权收盘价'))
_ret = _close_dividend.shift(-1) / _close_dividend - 1
_amount = stock_data.factor_clean(getattr(stock_data, '成交量（股）').rolling(21).sum())
_money = stock_data.factor_clean(getattr(stock_data, '成交额（元）').rolling(21).sum())

_close_train = _close.loc[begin_year:stop_year].dropna(axis = 1, how = 'all')
_open_train = _open.loc[begin_year:stop_year].dropna(axis = 1, how = 'all')
_high_train = _high.loc[begin_year:stop_year].dropna(axis = 1, how = 'all')
_low_train = _low.loc[begin_year:stop_year].dropna(axis = 1, how = 'all')
_ret_train = _ret.loc[begin_year:stop_year].dropna(axis = 1, how = 'all')
_amount_train = _amount.loc[begin_year:stop_year].dropna(axis = 1, how = 'all')
_money_train = _money.loc[begin_year:stop_year].dropna(axis = 1, how = 'all')

train_df = pd.DataFrame([_close_train.stack().rename('收盘价'),
                         _open_train.stack().rename('开盘价'),
                         _high_train.stack().rename('最高价'),
                         _low_train.stack().rename('最低价'),
                         _ret_train.stack().rename('收益率'),
                         _amount_train.stack().rename('成交量'),
                         _money_train.stack().rename('成交额'),
                         ]).T

train_df = train_df.reset_index(names = ['交易日期', '股票代码'])
train_X,train_Y, feature_names = make_XY(train_df, *different_axis)

'''
eval_df = pd.DataFrame([_close.loc[eval_begin_year:eval_stop_year, _close_train.columns].stack().rename('收盘价'),
                         _open.loc[eval_begin_year:eval_stop_year, _open_train.columns].stack().rename('开盘价'),
                         _high.loc[eval_begin_year:eval_stop_year, _high_train.columns].stack().rename('最高价'),
                         _low.loc[eval_begin_year:eval_stop_year, _low_train.columns].stack().rename('最低价'),
                         _ret.loc[eval_begin_year:eval_stop_year, _ret_train.columns].stack().rename('收益率'),
                         _amount.loc[eval_begin_year:eval_stop_year, _amount_train.columns].stack().rename('成交量'),
                         _money.loc[eval_begin_year:eval_stop_year, _money_train.columns].stack().rename('成交额'),

                         ]).T

eval_df = eval_df.reset_index(names = ['交易日期', '股票代码'])
eval_X,eval_Y, _ = make_XY(eval_df, *different_axis)
#%%
X = np.concatenate([train_X,eval_X],axis=0)
Y = np.concatenate([train_Y,eval_Y],axis=0)
X_feature_names = feature_names
sample_weight = []
sample_weight.extend([1]*train_X.shape[0])
sample_weight.extend([0]*eval_X.shape[0])
sample_weight = np.array(sample_weight)'''
X = np.concatenate([train_X],axis=0)
Y = np.concatenate([train_Y],axis=0)
X_feature_names = feature_names

sample_weight = []
sample_weight.extend([1]*train_X.shape[0])
sample_weight = np.array(sample_weight)

function_set_sample = ['common_add', 'common_sub', 'common_mul', 'common_div',
                       'common_log', 'common_sqrt', 'common_abs', 'common_inv', 'common_max', 'common_min', 'common_tan',]
my_function = [dynamic_ts_mean, dynamic_ts_max, dynamic_ts_std]
function_set = function_set_sample + my_function
# 这里的metric的填写是基于fitness.py 文件中的map里面的key值 时间问题这里就展示两个generation
gp_sample = genetic.SymbolicTransformer(generations=5,
                                        population_size=200,
                                        tournament_size=10,
                                        init_depth=(1, 3),
                                        hall_of_fame = 50,
                                        n_components=10,
                                        function_set=function_set,
                                        metric="pearson_3d",
                                        const_range=(-1, 1),
                                        p_crossover=0.4,
                                        p_hoist_mutation=0.001,
                                        p_subtree_mutation=0.01,
                                        p_point_mutation=0.01,
                                        p_point_replace=0.4,
                                        parsimony_coefficient="auto",
                                        feature_names=X_feature_names,
                                        max_samples=1, verbose=1,
                                        random_state=0, n_jobs=-2)

gp_sample.fit_3D(X, Y,feature_names,sample_weight=sample_weight,standard_expression="TRA ((pearson_3d>=0.02) and (spearman_3d >=0.002)) OOB (pearson_3d>0.0002)", need_parallel=True)

result = gp_sample.show_program(X,Y,sample_weight=sample_weight,feature_names=X_feature_names,baseIC=False,show_tracing=(True,"./show_tracing.xlsx"))
result.to_excel("./result_only10.xlsx")
