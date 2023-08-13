### 导入相关内容
import warnings
import os
import  numpy as np
import pandas as pd
from add_ts_function import ts_std_10, ts_max_10,  ts_mean_10
import genetic
import sys
sys.path.append(r'..')
from agency_portfolio.Portfolio_track import Portfolio_track
np.random.seed(10)
pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', True)
pd.set_option('display.unicode.ambiguous_as_wide', True)
warnings.filterwarnings('ignore')

# 常量定义
begin_year = pd.Timestamp('2017-01-01')
stop_year = pd.Timestamp('2019-01-01')

# 加载所有股票，这里为了演示就只加载了20只
stock_data = Portfolio_track()
_close = stock_data.to_MonthEnd(getattr(stock_data, '不复权收盘价').loc[begin_year:stop_year, :])
_open = stock_data.to_MonthEnd(getattr(stock_data, '不复权开盘价').loc[_close.index, _close.columns])
_high = stock_data.to_MonthEnd(getattr(stock_data, '不复权最高价').loc[_close.index,_close.columns])
_low = stock_data.to_MonthEnd(getattr(stock_data, '不复权最低价').loc[_close.index,_close.columns])
_close_dividend = stock_data.to_MonthEnd(getattr(stock_data, '复权收盘价'))
_ret = _close_dividend.shift(-1) / _close_dividend - 1
_ret = _ret.loc[_close.index,_close.columns]

total_df = pd.DataFrame([_close.stack().rename('收盘价'), _open.stack().rename('开盘价'),
                         _high.stack().rename('最高价'), _low.stack().rename('最低价'),
                         _ret.stack().rename('收益率')
                         ]).T
total_df = total_df.reset_index(names = ['交易日期', '股票代码'])

total_df = total_df[["股票代码","交易日期","开盘价","最高价","最低价","收盘价", "收益率"]]
print(len(total_df["股票代码"].unique()))

#二位转三维的函数
def make_XY(df, index_name, columns_name, Y_column,):
    '''

    Args:
        df: 输入的dataFrame shape:[trade_dates * stocks,features] [stock000001(trade_dates,features),stock000002(trade_dates,features)...]
        index_name: 交易日期
        columns_name: 股票名称
        Y_column1: 预测的对象

    Returns: X,Y,feature_names

    '''
    df = df.pivot_table(index=[index_name], columns=[columns_name], sort=True, dropna=False)
    Y1 = df.loc[:,(Y_column,)].to_numpy(dtype=np.double)

    df = df.drop([Y_column,],axis=1)
    X_0_len = len(df.index)

    X_1_len = len(df.columns.levels[0]) - 1
    X_2_len = len(df.columns.levels[1])
    return df.to_numpy(dtype=np.double).reshape((X_0_len, X_1_len, X_2_len)), Y1, df.columns.levels[0].drop([Y_column,])

X,Y1, X_feature_names = make_XY(total_df, "交易日期","股票代码", "收益率",)
function_set_sample = ['common_add', 'common_sub', 'common_mul', 'common_div',
                       'common_log', 'common_sqrt', 'common_abs', 'common_inv', 'common_max', 'common_min', 'common_tan',] #'std_10'
my_function = [ts_std_10, ts_max_10,  ts_mean_10,]
function_set = function_set_sample + my_function

print(total_df.shape)
print(X.shape)
print(Y1.shape)

# 这里的metric的填写是基于fitness.py 文件中的map里面的key值 时间问题这里就展示两个generation
gp_sample = genetic.SymbolicTransformer(generations=2,
                                        population_size=200,
                                        tournament_size=10,
                                        init_depth=(1, 3),
                                        hall_of_fame=100,
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

sample_weight = np.array([1]*len(X))
gp_sample.fit_3D(X, Y1,feature_names=X_feature_names,sample_weight=sample_weight,need_parallel=True)

result = gp_sample.show_program(X,Y1,X_feature_names,baseIC=False)
result