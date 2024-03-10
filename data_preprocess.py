
import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import talib as tb
from talib import abstract
from sklearn.tree import export_graphviz
import graphviz
import os
from graphviz import Digraph
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties


from sklearn.ensemble import RandomForestClassifier

# DLModel
import tensorflow as tf
from tensorflow import keras


import warnings

warnings.filterwarnings('ignore')





#　===================================================================

class DataForPredict:
    def __init__(self, price_data_path, eco_data_path, anno_data_path, daily_data_path, daily_data_path2):
        self.price_data = pd.read_excel(price_data_path)
        self.eco_data = pd.read_excel(eco_data_path, sheet_name='策略用數據')
        self.anno_data = pd.read_excel(anno_data_path, index_col=0)
        self.daily_data = pd.read_excel(daily_data_path)
        index_place_1 = self.daily_data.columns.get_loc('Gold Price')
        for i in range(index_place_1, self.daily_data.shape[1]):
            self.daily_data[self.daily_data.columns[i]].fillna(method='ffill', inplace=True)
        for i in range(index_place_1, self.daily_data.shape[1]):
            self.daily_data[self.daily_data.columns[i]].fillna(value=0, inplace=True)

        self.ta = None
        self.merged_data = None
        self.o_col_num = None




def change_df(df):
    arr = df.values
    new_df = pd.DataFrame(arr[1:, 1:], index=arr[1:, 0], columns=arr[0, 1:])
    new_df.index.name = arr[0, 0]
    return new_df


def del_price(data_for_predict):
    data_for_predict.price_data = data_for_predict.change_df(data_for_predict.price_data)
    data_for_predict.price_data.reset_index(inplace=True)


    data_for_predict.price_data['year'] = pd.DatetimeIndex(data_for_predict.price_data['交易日期']).year
    data_for_predict.price_data = data_for_predict.price_data.loc[data_for_predict.price_data['year'] >= 2005]
    data_for_predict.price_data.drop('year', axis=1, inplace=True)
    data_for_predict.price_data.reset_index(drop=True, inplace=True)
    # .drop(data_for_predict.price_data.index[[0]], inplace=True)

def del_ta_lib(data_for_predict):

    # 只留開高低收量，把日期變成index
    data_for_predict.price_data.set_index('交易日期', inplace=True)
    data_for_predict.price_data = pd.DataFrame(data_for_predict.price_data, columns=['開盤價', '最高價', '最低價', '收盤價', '成交量'])
    # data_for_predict.price_data = pd.DataFrame(data_for_predict.price_data, columns=['交易日期', '收盤價'])

    # 改成 TA-Lib 可以辨識的欄位名稱
    data_for_predict.price_data.columns = ['open', 'high', 'low', 'close', 'volume']
    data_for_predict.price_data.drop(data_for_predict.price_data[data_for_predict.price_data['volume'] == "-"].index, axis=0, inplace=True)
    # 計算技術指標
    o = data_for_predict.price_data['open'].astype(float).values
    c = data_for_predict.price_data['close'].astype(float).values
    h = data_for_predict.price_data['high'].astype(float).values
    l = data_for_predict.price_data['low'].astype(float).values
    v = data_for_predict.price_data['volume'].astype(float).values

    data_for_predict.ta = pd.DataFrame(index=data_for_predict.price_data.index)

    data_for_predict.ta['MA10'] = tb.MA(c, timeperiod=10)

    data_for_predict.ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
    data_for_predict.ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
    data_for_predict.ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]

    data_for_predict.ta['k1'], data_for_predict.ta['d1'] = tb.STOCH(h, l, c, fastk_period=5, slowk_period=3)


    data_for_predict.price_data = pd.merge(data_for_predict.price_data, data_for_predict.ta,
                              left_on=data_for_predict.price_data.index, right_on=data_for_predict.ta.index)
    data_for_predict.price_data = data_for_predict.price_data.set_index('key_0')

    ta_list = ['MACD', 'RSI', 'MOM', 'STOCH']
    # 快速計算與整理因子
    for x in ta_list:
        output = eval('abstract.' + x + '(data_for_predict.price_data)')
        output.name = x.lower() if type(output) == pd.core.series.Series else None
        data_for_predict.price_data = pd.merge(data_for_predict.price_data, pd.DataFrame(output), left_on=data_for_predict.price_data.index,
                                  right_on=output.index)
        data_for_predict.price_data = data_for_predict.price_data.set_index('key_0')

    data_for_predict.price_data.rename(columns={'close': 'Mini-S&P500收盤價'}, inplace=True)
    data_for_predict.price_data.reset_index(inplace=True)
    data_for_predict.price_data.rename(columns={'key_0': '交易日期'}, inplace=True)

    data_for_predict.merged_data = data_for_predict.price_data.copy()

def mer_daily_data(data_for_predict):
    data_for_predict.daily_data['交易日期'] = pd.to_datetime(data_for_predict.daily_data['交易日期'])
    data_for_predict.merged_data.dropna(inplace=True, axis=0)
    data_for_predict.merged_data = data_for_predict.merged_data.merge(data_for_predict.daily_data, how='inner', on=['交易日期'])
    data_for_predict.o_col_num = data_for_predict.merged_data.shape[1]

def del_eco(data_for_predict):
    data_for_predict.eco_data = data_for_predict.change_df(data_for_predict.eco_data)
    per1 = pd.date_range(start='1-1-2005',
                         end='06-30-2022', freq='M')
    data_for_predict.eco_data.index = per1
    data_for_predict.eco_data.reset_index(inplace=True)
    data_for_predict.eco_data.rename(columns={'index': 'Date'}, inplace=True)
    data_for_predict.eco_data['year'] = pd.DatetimeIndex(data_for_predict.eco_data['Date']).year
    data_for_predict.eco_data['month'] = pd.DatetimeIndex(data_for_predict.eco_data['Date']).month
    data_for_predict.eco_data.iloc[:, 0] = data_for_predict.eco_data.iloc[:, 0].apply(lambda _: datetime.strftime(_, "%Y/%m"))


def del_anno(data_for_predict):
    per1 = pd.date_range(start='01-01-2005',
                         end='7-31-2022', freq='M')
    data_for_predict.anno_data.index = per1
    data_for_predict.anno_data.reset_index(inplace=True)
    data_for_predict.anno_data.rename(columns={'index': 'Date'}, inplace=True)
    data_for_predict.anno_data.iloc[:, 0] = data_for_predict.anno_data.iloc[:, 0].apply(lambda _: datetime.strftime(_, "%Y%m%d"))

    fill_date = data_for_predict.anno_data.iloc[:, 0]

    for i in range(1, data_for_predict.anno_data.shape[1]):
        data_for_predict.anno_data.iloc[:, i].fillna(value=fill_date, inplace=True)

    for i in range(1, data_for_predict.anno_data.shape[1]):
        data_for_predict.anno_data.iloc[:, i] = data_for_predict.anno_data.iloc[:, i].astype(str)
        data_for_predict.anno_data.iloc[:, i] = data_for_predict.anno_data.iloc[:, i].apply(lambda _: _[:4] + "/" + _[4:6] + "/" + _[6:8])
        data_for_predict.anno_data.iloc[:, i] = data_for_predict.anno_data.iloc[:, i].apply(lambda _: datetime.strptime(_, "%Y/%m/%d"))

    data_for_predict.anno_data.drop([data_for_predict.anno_data.columns[0]], axis=1, inplace=True)
    data_for_predict.anno_data.index = per1
    data_for_predict.anno_data.reset_index(inplace=True)
    data_for_predict.anno_data.rename(columns={'index': 'Date'}, inplace=True)
    data_for_predict.anno_data['year'] = pd.DatetimeIndex(data_for_predict.anno_data['Date']).year
    data_for_predict.anno_data['month'] = pd.DatetimeIndex(data_for_predict.anno_data['Date']).month
    data_for_predict.anno_data.iloc[:, 0] = data_for_predict.anno_data.iloc[:, 0].apply(lambda _: datetime.strftime(_, "%Y/%m"))

def creat_eco_df(data_for_predict):
    data_for_predict.merged_data['day'] = pd.DatetimeIndex(data_for_predict.merged_data['交易日期']).day
    data_for_predict.merged_data['year'] = pd.DatetimeIndex(data_for_predict.merged_data['交易日期']).year
    data_for_predict.merged_data['month'] = pd.DatetimeIndex(data_for_predict.merged_data['交易日期']).month
    for i in range(1, data_for_predict.anno_data.shape[1] - 3):
        tgt_name = data_for_predict.anno_data.columns[i]
        data_for_predict.merged_data.insert(data_for_predict.merged_data.shape[1], tgt_name, value=np.nan)
        for year in range(2005, 2023):
            for month in range(1, 13):
                if year == 2022:
                    if month > 7:
                        continue
                if month == 1:
                    if year == 2005:
                        continue
                    sel_eco_b = data_for_predict.eco_data[(data_for_predict.eco_data['year'] == year - 1) & (data_for_predict.eco_data['month'] == 11)]
                    eco_b = sel_eco_b[tgt_name].iloc[0]
                    sel_eco = data_for_predict.eco_data[(data_for_predict.eco_data['year'] == year - 1) & (data_for_predict.eco_data['month'] == 12)]
                    eco = sel_eco[tgt_name].iloc[0]

                elif month == 2:
                    if year == 2005:
                        continue
                    sel_eco_b = data_for_predict.eco_data[(data_for_predict.eco_data['year'] == year - 1) & (data_for_predict.eco_data['month'] == 12)]
                    eco_b = sel_eco_b[tgt_name].iloc[0]
                    sel_eco = data_for_predict.eco_data[(data_for_predict.eco_data['year'] == year) & (data_for_predict.eco_data['month'] == month - 1)]
                    eco = sel_eco[tgt_name].iloc[0]
                else:
                    sel_eco_b = data_for_predict.eco_data[(data_for_predict.eco_data['year'] == year) & (data_for_predict.eco_data['month'] == month - 2)]
                    eco_b = sel_eco_b[tgt_name].iloc[0]
                    sel_eco = data_for_predict.eco_data[(data_for_predict.eco_data['year'] == year) & (data_for_predict.eco_data['month'] == month - 1)]
                    eco = sel_eco[tgt_name].iloc[0]

                sel_anno = data_for_predict.anno_data[(data_for_predict.anno_data['year'] == year) & (data_for_predict.anno_data['month'] == month)]
                anno_date = pd.DatetimeIndex(sel_anno[tgt_name]).day[0]
                data_for_predict.merged_data.insert(data_for_predict.merged_data.shape[1], 'anno_date', value=anno_date)

                data_for_predict.merged_data.loc[(data_for_predict.merged_data['year'] == year) &
                                    (data_for_predict.merged_data['month'] == month) &
                                    (data_for_predict.merged_data['day'] < data_for_predict.merged_data['anno_date'])] \
                    = data_for_predict.merged_data.loc[(data_for_predict.merged_data['year'] == year) &
                                          (data_for_predict.merged_data['month'] == month) &
                                          (data_for_predict.merged_data['day'] < data_for_predict.merged_data['anno_date'])] \
                    .fillna(value=eco_b)

                data_for_predict.merged_data.loc[(data_for_predict.merged_data['year'] == year) &
                                    (data_for_predict.merged_data['month'] == month) &
                                    (data_for_predict.merged_data['day'] >= data_for_predict.merged_data['anno_date'])] \
                    = data_for_predict.merged_data.loc[(data_for_predict.merged_data['year'] == year) &
                                          (data_for_predict.merged_data['month'] == month) &
                                          (data_for_predict.merged_data['day'] >= data_for_predict.merged_data['anno_date'])] \
                    .fillna(value=eco)

                data_for_predict.merged_data.drop('anno_date', axis=1, inplace=True)
    data_for_predict.merged_data.dropna(axis=0, inplace=True)
    data_for_predict.merged_data.drop('day', axis=1, inplace=True)
    data_for_predict.merged_data.drop('year', axis=1, inplace=True)
    data_for_predict.merged_data.drop('month', axis=1, inplace=True)

# main
data_for_predict = DataForPredict(r"Mini_SP500_price.xlsx",
                          r"eco_data_month.xlsx",
                          r"anno_data.xlsx",
                          r'eco_data_daily.xlsx',
                          r'eco_data_daily_index.xlsx')
del_price(data_for_predict)
del_ta_lib(data_for_predict)
mer_daily_data(data_for_predict)
del_eco(data_for_predict)
del_anno(data_for_predict)
creat_eco_df(data_for_predict)
data_for_predict.merged_data.to_excel("merged_data.xlsx", index=False)

