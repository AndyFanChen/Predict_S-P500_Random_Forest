import pandas as pd
import numpy as np
import datetime
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import talib as tb
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class DataPreprocess:
    def __init__(self, price_data_path, eco_data_path, anno_data_path, daily_data_path, daily_data_path2):
        self.price_data = pd.read_excel(price_data_path)
        self.eco_data = pd.read_excel(eco_data_path, sheet_name='策略用數據')
        self.anno_data = pd.read_excel(anno_data_path, index_col=0)
        self.daily_data = pd.read_excel(daily_data_path)
        self.daily_data2 = pd.read_excel(daily_data_path2)
        index_place1 = self.daily_data.columns.get_loc('Gold Price')
        for i in range(index_place1, self.daily_data.shape[1]):
            self.daily_data[self.daily_data.columns[i]].fillna(method='ffill', inplace=True)
        for i in range(index_place1, self.daily_data.shape[1]):
            self.daily_data[self.daily_data.columns[i]].fillna(value=0, inplace=True)
        for i in range(self.daily_data2.shape[1]):
            self.daily_data2[self.daily_data2.columns[i]].fillna(method='ffill', inplace=True)
        for i in range(self.daily_data2.shape[1]):
            self.daily_data2[self.daily_data2.columns[i]].fillna(value=0, inplace=True)
        self.ta = None
        self.merged_data = None
        self.o_col_num = None

    def change_df(self, df):
        arr = df.values
        new_df = pd.DataFrame(arr[1:, 1:], index=arr[1:, 0], columns=arr[0, 1:])
        new_df.index.name = arr[0, 0]
        return new_df

    def del_price(self):
        self.price_data = self.change_df(self.price_data)
        self.price_data.reset_index(inplace=True)
        self.price_data['year'] = pd.DatetimeIndex(self.price_data['交易日期']).year
        self.price_data = self.price_data.loc[self.price_data['year'] >= 2005]
        self.price_data.drop('year', axis=1, inplace=True)
        self.price_data.reset_index(drop=True, inplace=True)

    def del_talib(self):

        # 只留開高低收量，把日期變成index
        self.price_data.set_index('交易日期', inplace=True)
        self.price_data = pd.DataFrame(self.price_data, columns=['開盤價', '最高價', '最低價', '收盤價', '成交量'])

        # 改成 TA-Lib 可以辨識的欄位名稱
        self.price_data.columns = ['open', 'high', 'low', 'close', 'volume']
        self.price_data.drop(self.price_data[self.price_data['volume'] == "-"].index, axis=0, inplace=True)
        # 計算技術指標
        o = self.price_data['open'].astype(float).values
        c = self.price_data['close'].astype(float).values
        h = self.price_data['high'].astype(float).values
        l = self.price_data['low'].astype(float).values
        v = self.price_data['volume'].astype(float).values

        self.ta = pd.DataFrame(index=self.price_data.index)
        # self.ta['MA5'] = tb.MA(c, timeperiod=5)
        # print(self.ta)
        self.ta['MA10'] = tb.MA(c, timeperiod=10)
        self.ta['MA20'] = tb.MA(c, timeperiod=20)
        self.ta['MA16'] = tb.MA(c, timeperiod=16)

        self.ta['ADX'] = tb.ADX(h, l, c, timeperiod=14)
        self.ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14)

        self.ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0]
        self.ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1]
        self.ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2]

        self.ta['MACD1'], self.ta['MACDSignal1'], self.ta['MACDhist1'] = tb.MACD(c, fastperiod=10, slowperiod=22,
                                                                                 signalperiod=7)
        for i in range(2, 10):
            self.ta['MACD{}'.format(i)], self.ta['MACDSignal{}'.format(i)], self.ta['MACDhist{}'.format(i)] = tb.MACD(c, fastperiod=10 + i, slowperiod=22 + i,
                                                                                     signalperiod=7 + i)

        self.ta['MOM1'] = tb.MOM(c, timeperiod=7)
        self.ta['MOM2'] = tb.MOM(c, timeperiod=11)
        self.ta['MOM3'] = tb.MOM(c, timeperiod=13)
        for i in range(4, 10):
            self.ta['MOM{}'.format(i)] = tb.MOM(c, timeperiod=10 + i)

        self.ta['k1'], self.ta['d1'] = tb.STOCH(h, l, c, fastk_period=5, slowk_period=3)
        self.ta['k2'], self.ta['d2'] = tb.STOCH(h, l, c, fastk_period=11, slowk_period=4)
        self.ta['k3'], self.ta['d3'] = tb.STOCH(h, l, c, fastk_period=14, slowk_period=7)
        for i in range(4, 10):
            self.ta['k{}'.format(i)], self.ta['d{}'.format(i)] = tb.STOCH(h, l, c, fastk_period=12 + i*2, slowk_period=8+i)

        self.ta['AD'] = tb.AD(h, l, c, v)
        self.ta['ATR'] = tb.ATR(h, l, c, timeperiod=14)
        self.ta['HT_DC'] = tb.HT_DCPERIOD(c)

        self.price_data = pd.merge(self.price_data, self.ta,
                                  left_on=self.price_data.index, right_on=self.ta.index)
        self.price_data = self.price_data.set_index('key_0')

        ta_list = ['MACD', 'RSI', 'MOM', 'STOCH']
        # 快速計算與整理因子
        for x in ta_list:
            output = eval('abstract.' + x + '(self.price_data)')
            output.name = x.lower() if type(output) == pd.core.series.Series else None
            self.price_data = pd.merge(self.price_data, pd.DataFrame(output), left_on=self.price_data.index,
                                      right_on=output.index)
            self.price_data = self.price_data.set_index('key_0')

        self.price_data.rename(columns={'close': 'Mini-S&P500收盤價'}, inplace=True)
        self.price_data.reset_index(inplace=True)
        self.price_data.rename(columns={'key_0': '交易日期'}, inplace=True)

        self.merged_data = self.price_data.copy()

    def mer_daily_data(self):
        self.daily_data['交易日期'] = pd.to_datetime(self.daily_data['交易日期'])
        self.merged_data.dropna(inplace=True, axis=0)
        self.merged_data = self.merged_data.merge(self.daily_data, how='inner', on=['交易日期'])
        self.o_col_num = self.merged_data.shape[1]

    def mer_daily_data2(self):
        self.daily_data2['交易日期'] = pd.to_datetime(self.daily_data2['交易日期'])
        self.merged_data.dropna(inplace=True, axis=0)
        self.merged_data = self.merged_data.merge(self.daily_data2, how='inner', on=['交易日期'])
        self.o_col_num = self.merged_data.shape[1]


    def del_eco(self):
        self.eco_data = self.change_df(self.eco_data)
        per1 = pd.date_range(start='1-1-2005',
                             end='06-30-2022', freq='M')
        self.eco_data.index = per1
        self.eco_data.reset_index(inplace=True)
        self.eco_data.rename(columns={'index': 'Date'}, inplace=True)
        self.eco_data['year'] = pd.DatetimeIndex(self.eco_data['Date']).year
        self.eco_data['month'] = pd.DatetimeIndex(self.eco_data['Date']).month
        self.eco_data.iloc[:, 0] = self.eco_data.iloc[:, 0].apply(lambda _: datetime.strftime(_, "%Y/%m"))
        # self.eco_data.drop(['公佈日期'], axis=1, inplace=True)
        # self.eco_data.dropna(axis=0, inplace=True)

    def del_anno(self):
        per1 = pd.date_range(start='01-01-2005',
                             end='7-31-2022', freq='M')
        self.anno_data.index = per1
        self.anno_data.reset_index(inplace=True)
        self.anno_data.rename(columns={'index': 'Date'}, inplace=True)
        self.anno_data.iloc[:, 0] = self.anno_data.iloc[:, 0].apply(lambda _: datetime.strftime(_, "%Y%m%d"))

        fill_date = self.anno_data.iloc[:, 0]

        for i in range(1, self.anno_data.shape[1]):
            self.anno_data.iloc[:, i].fillna(value=fill_date, inplace=True)

        for i in range(1, self.anno_data.shape[1]):
            self.anno_data.iloc[:, i] = self.anno_data.iloc[:, i].astype(str)
            self.anno_data.iloc[:, i] = self.anno_data.iloc[:, i].apply(lambda _: _[:4] + "/" + _[4:6] + "/" + _[6:8])
            self.anno_data.iloc[:, i] = self.anno_data.iloc[:, i].apply(lambda _: datetime.strptime(_, "%Y/%m/%d"))

        self.anno_data.drop([self.anno_data.columns[0]], axis=1, inplace=True)
        self.anno_data.index = per1
        self.anno_data.reset_index(inplace=True)
        self.anno_data.rename(columns={'index': 'Date'}, inplace=True)
        self.anno_data['year'] = pd.DatetimeIndex(self.anno_data['Date']).year
        self.anno_data['month'] = pd.DatetimeIndex(self.anno_data['Date']).month
        self.anno_data.iloc[:, 0] = self.anno_data.iloc[:, 0].apply(lambda _: datetime.strftime(_, "%Y/%m"))

    def creat_eco_df(self):
        self.merged_data['day'] = pd.DatetimeIndex(self.merged_data['交易日期']).day
        self.merged_data['year'] = pd.DatetimeIndex(self.merged_data['交易日期']).year
        self.merged_data['month'] = pd.DatetimeIndex(self.merged_data['交易日期']).month
        for i in range(1, self.anno_data.shape[1] - 3):
            tgt_name = self.anno_data.columns[i]
            self.merged_data.insert(self.merged_data.shape[1], tgt_name, value=np.nan)
            for year in range(2005, 2023):
                for month in range(1, 13):
                    if year == 2022:
                        if month > 7:
                            continue
                    if month == 1:
                        if year == 2005:
                            continue
                        sel_eco_b = self.eco_data[(self.eco_data['year'] == year - 1) & (self.eco_data['month'] == 11)]
                        eco_b = sel_eco_b[tgt_name].iloc[0]
                        sel_eco = self.eco_data[(self.eco_data['year'] == year - 1) & (self.eco_data['month'] == 12)]
                        eco = sel_eco[tgt_name].iloc[0]

                    elif month == 2:
                        if year == 2005:
                            continue
                        sel_eco_b = self.eco_data[(self.eco_data['year'] == year - 1) & (self.eco_data['month'] == 12)]
                        eco_b = sel_eco_b[tgt_name].iloc[0]
                        sel_eco = self.eco_data[(self.eco_data['year'] == year) & (self.eco_data['month'] == month - 1)]
                        eco = sel_eco[tgt_name].iloc[0]
                    else:
                        sel_eco_b = self.eco_data[(self.eco_data['year'] == year) & (self.eco_data['month'] == month - 2)]
                        eco_b = sel_eco_b[tgt_name].iloc[0]
                        sel_eco = self.eco_data[(self.eco_data['year'] == year) & (self.eco_data['month'] == month - 1)]
                        eco = sel_eco[tgt_name].iloc[0]

                    sel_anno = self.anno_data[(self.anno_data['year'] == year) & (self.anno_data['month'] == month)]
                    anno_date = pd.DatetimeIndex(sel_anno[tgt_name]).day[0]
                    self.merged_data.insert(self.merged_data.shape[1], 'anno_date', value=anno_date)

                    self.merged_data.loc[(self.merged_data['year'] == year) &
                                        (self.merged_data['month'] == month) &
                                        (self.merged_data['day'] < self.merged_data['anno_date'])] \
                        = self.merged_data.loc[(self.merged_data['year'] == year) &
                                              (self.merged_data['month'] == month) &
                                              (self.merged_data['day'] < self.merged_data['anno_date'])] \
                        .fillna(value=eco_b)

                    self.merged_data.loc[(self.merged_data['year'] == year) &
                                        (self.merged_data['month'] == month) &
                                        (self.merged_data['day'] >= self.merged_data['anno_date'])] \
                        = self.merged_data.loc[(self.merged_data['year'] == year) &
                                              (self.merged_data['month'] == month) &
                                              (self.merged_data['day'] >= self.merged_data['anno_date'])] \
                        .fillna(value=eco)

                    self.merged_data.drop('anno_date', axis=1, inplace=True)
        self.merged_data.dropna(axis=0, inplace=True)
        self.merged_data.drop('day', axis=1, inplace=True)
        self.merged_data.drop('year', axis=1, inplace=True)
        self.merged_data.drop('month', axis=1, inplace=True)


# %%
class MLTrainEval:

    def __init__(self, merged_data):
        self.merged_data = merged_data
        self.merged_data.fillna(method='ffill', axis=0, inplace=True)
        self.cost_count = 0
        self.sp500_profit = None
        self.importance = None
        self.confusion_matrix = None
        self.oob_score = None
        self.score = None
        self.test = None
        self.confusion_matrix_train = None
        self.profit_list = []
        self.per_profit_list = []

    def creat_y(self):
        # shift(-1)代表今天的與明天的比
        self.merged_data['week_trend'] = np.where(
            self.merged_data['open'].shift(-2) > self.merged_data['open'].shift(-1), 1, 0)
        self.merged_data.dropna(inplace=True, axis=0)

    def data_split(self):
        split_point = int(self.merged_data.shape[0] * 0.75)
        # 切割成學習樣本以及測試樣本
        train = self.merged_data.iloc[:split_point, 1:].copy()

        test = self.merged_data.iloc[split_point:-1, 1:].copy()
        for i in range(train.shape[1]):
            # print(i)
            train[train.columns[i]] = pd.to_numeric(train[train.columns[i]])
        for i in range(test.shape[1]):
            test[test.columns[i]] = pd.to_numeric(test[test.columns[i]])

        train_X = train.drop('week_trend', axis=1)
        train_X.dropna(inplace=True, axis=0)

        train_y = train.week_trend
        # 測試樣本再分成目標序列 y 以及因子矩陣 X
        test_X = test.drop('week_trend', axis=1)
        test_X.dropna(inplace=True, axis=0)
        test_y = test.week_trend

        return train_X, train_y, test_X, test_y, test

    # 優化參數(可用CV或oob score)
    # 法一：oob score

    def parm_optim_oob(self, train_X, train_y):
        max_features = round(np.sqrt(self.merged_data.shape[1] - 1))

        best_oob = 0
        n_tree_list = [1000]
        max_depth_list = [1, 2, 3, 4, 5, 10, 20, 40, 60, 100]
        min_samples_leaf_list = [1, 2, 3, 4, 5, 10, 20, 40, 60, 100]
        best_list =[]
        for n_tree in n_tree_list:
            # print(n_tree)
            for max_depth in range(1, 100, 2):
                print(max_depth)
                best_oob_this = 0
                for min_samples_leaf in range(1, 60, 4):
                    rf = RandomForestClassifier(n_estimators=n_tree, n_jobs=-1, oob_score=True, random_state=42,
                                                max_features=max_features, max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf)
                    rf.fit(train_X, train_y)
                    # training_score = rf.score(train_X, train_y)
                    # print(training_score)

                    oob_score = rf.oob_score_
                    if oob_score > best_oob_this:
                        best_oob_this = oob_score
                        all_list_this = [n_tree, max_depth, min_samples_leaf]
                        print('best_oob_this')
                        print(best_oob_this, all_list_this)
                    all_list = [n_tree, max_depth, min_samples_leaf]
                    print(oob_score, all_list)
                    if oob_score > best_oob:
                        best_list = [n_tree, max_depth, min_samples_leaf]
                        best_oob = oob_score
                        print("the best")
                        print(best_oob, best_list)
        return best_oob, best_list




    def ml_decision(self, train_X, train_y, test_X, n_tree, max_depth=None, min_samples_leaf=1):
        # Instantiate model with 1000 decision trees
        max_features = round(np.sqrt(self.merged_data.shape[1] - 1))
        # max_features = self.merged_data.shape[1]
        rf = RandomForestClassifier(n_estimators=n_tree, n_jobs=-1, random_state=42,
                                    max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        rf.fit(train_X, train_y)
        feature_names = train_X.columns
        # 印出各因子重要程度
        importances = rf.feature_importances_
        forest_importances = pd.Series(importances, index=feature_names).sort_values()
        self.importance = forest_importances
        print(forest_importances)
        prediction_training = rf.predict(train_X)
        training_score = rf.score(train_X, train_y)
        self.confusion_matrix_train = confusion_matrix(train_y, prediction_training)
        prediction = rf.predict(test_X)

        return rf, prediction, training_score

    def view_result(self, rf, train_X, test_X, test_y, prediction):

        # 混淆矩陣
        self.confusion_matrix = confusion_matrix(test_y, prediction)
        # print(confusion_matrix)

        # 準確率
        self.score = rf.score(test_X, test_y)
        # print("test組分數")
        # print(self.score)

    def str_view(self, prediction):
        split_point = int(self.merged_data.shape[0] * 0.75)
        self.test = self.merged_data.iloc[split_point:-1, :].copy()
        self.test['prediction'] = prediction

    def profit_cal(self):

        # 每天進出場
        self.profit_list = []
        self.test.reset_index(drop=True, inplace=True)
        self.test.insert(self.test.shape[1], 'IfTrade', value="")
        self.test.insert(self.test.shape[1], 'profit', value="")
        cost = 0.5
        self.cost_count = 0
        for i in range(self.test.shape[0]):
            try:

                short_sig = self.test['prediction'].iloc[i - 1]
                if self.test['prediction'].iloc[i] == 1 and self.test['prediction'].iloc[i - 1] != 0 and self.test['prediction'].iloc[i - 2] != 0:

                    in_price = self.test['open'].iloc[i + 1]
                    out_price = self.test['open'].iloc[i + 2]
                    self.test['IfTrade'].iloc[i + 1] = "Long"
                    profit = out_price - in_price
                    # 算成本
                    if i == 0:
                        profit -= cost
                        self.cost_count += 1
                    else:
                        if self.test['IfTrade'].iloc[i] != "Long":
                            profit -= cost
                            self.cost_count += 1
                    self.test['profit'].iloc[i + 1] = profit
                    self.profit_list.append(profit)
                    per_profit = profit / in_price
                    self.per_profit_list.append(per_profit)
                if self.test['IfTrade'].iloc[i] == "":
                    self.test['profit'].iloc[i + 1] = 0
                    self.per_profit_list.append(0)
            except IndexError:
                pass

        np_profit = np.array(self.profit_list)
        profit = np_profit.sum()
        profit_std = np_profit.std()
        np_per_profit = np.array(self.per_profit_list)
        per_profit = np_per_profit.mean() * 252

        # 買進持有損益
        self.test['open損益'] = self.test['open'].shift(-1) - \
                                        self.test['open']
        self.sp500_profit = self.test['open損益'].to_numpy()
        sum_list_sp = []
        accumulate_sp = 0
        for i in range(self.sp500_profit.shape[0]):
            accumulate_sp += self.sp500_profit[i]
            sum_list_sp.append(accumulate_sp)
        sp500_profit_sum = self.test['open'].iloc[-1] - self.test['open'].iloc[0]
        sp500_profit_std = self.sp500_profit.std()
        per_sp500_profit = ((self.test['open'].iloc[-1] / self.test['open'].iloc[0] - 1) / self.sp500_profit.shape[0]) * 252



        return profit, per_profit, profit_std, sp500_profit_sum, sp500_profit_std, per_sp500_profit

    def plot_loss(self, dpi=500):
        # 策略損益
        trade_date = self.test.loc[(self.test['IfTrade'] == "Long") | (self.test['IfTrade'] == "Short")]['交易日期']
        sum_list = []
        accumulate = 0
        for i in range(len(self.profit_list)):
            accumulate += self.profit_list[i]
            sum_list.append(accumulate)

        fig, ax = plt.subplots()
        font_set = FontProperties(fname=r"c:\windows\fonts\msjh.ttc", size=15, weight='bold')
        ax.plot(trade_date, sum_list, color='forestgreen')
        fig.autofmt_xdate()

        sum_list_sp = []
        accumulate_sp = 0
        for i in range(self.sp500_profit.shape[0]):
            accumulate_sp += self.sp500_profit[i]
            sum_list_sp.append(accumulate_sp)

        title = '策略與買進持有累積報酬絕對值比較'
        plt.title(title, fontproperties=font_set)
        ax.plot(self.test['交易日期'], sum_list_sp, color='forestgreen', alpha=0.4555555)
        font_set2 = FontProperties(fname=r"c:\windows\fonts\msjh.ttc", size=10, weight='bold')
        plt.xlabel('日期', fontproperties=font_set2)
        plt.ylabel('損益(點)', fontproperties=font_set2)
        font_set3 = FontProperties(fname=r"c:\windows\fonts\msjh.ttc", size=9, weight='bold')
        plt.legend(['機器學習策略報酬(點)', '買進持有報酬(點)'], prop=font_set3)
        fig.autofmt_xdate()
        plt.savefig('策略與買進持有累積報酬絕對值比較(隨機森林)', dpi=dpi)

        plt.show()


# main
# %%
sp500_path = ""
month_macro_eco_path = ""
data_date = ""
day_macro_eco_data = ""
firm_data = ""

data_preprocess = DataPreprocess(sp500_path,
                          month_macro_eco_path,
                          data_date,
                          day_macro_eco_data,
                          firm_data)
data_preprocess.del_price()
data_preprocess.del_talib()
data_preprocess.mer_daily_data()
# data_preprocess.mer_daily_data2()
data_preprocess.del_eco()
data_preprocess.del_anno()
data_preprocess.creat_eco_df()

#%%
merged_data_path = ""
data_preprocess.merged_data.to_excel(merged_data_path, index=False)
merged_data = pd.read_excel(merged_data_path)

# %%
merged_data_shift = merged_data.copy()
merged_data_shift.iloc[:, 5:] = merged_data_shift.iloc[:, 5:] - merged_data_shift.iloc[:, 5:].shift(1)
merged_data_shift.drop(columns=merged_data_shift.columns[1:6].values, axis=1, inplace=True)
merged_data_new = merged_data.merge(merged_data_shift, how='inner', on=['交易日期'])

#%%
# ml_train_eval = MLTrainEval(data_preprocess.merged_data)
ml_train_eval = MLTrainEval(merged_data_new)
# ml_train_eval = MLTrainEval(merged_data)
ml_train_eval.creat_y()
train_X, train_y, test_X, test_y, test = ml_train_eval.data_split()
#%%
# 找最佳參數
best_oob, best_list = ml_train_eval.parm_optim_oob(train_X, train_y)

#%%

n_tree = 300
max_depth = 4
min_samples_leaf = 9
rf, prediction, training_score = ml_train_eval.ml_decision(train_X, train_y, test_X, n_tree=n_tree, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
ml_train_eval.view_result(rf, train_X, test_X, test_y, prediction)
print("treeNum：{} max_depth：{} min_samples_leaf：{}　training_score：{} testingScore：{}"
      .format(n_tree, max_depth, min_samples_leaf, training_score, ml_train_eval.score))
print('Short Rate')
print(ml_train_eval.confusion_matrix[0, 0] / (ml_train_eval.confusion_matrix[0, 0] + ml_train_eval.confusion_matrix[1, 0]))
print('Long Rate')
print(print(ml_train_eval.confusion_matrix[1, 1] / (ml_train_eval.confusion_matrix[0, 1] + ml_train_eval.confusion_matrix[1, 1])))


ml_train_eval.view_result(rf, train_X, test_X, test_y, prediction)
ml_train_eval.str_view(prediction)
profit, per_profit, profit_std, sp500_profit_sum, sp500_profit_std, per_sp500_profit = ml_train_eval.profit_cal()

ml_train_eval.plot_loss(dpi=500)