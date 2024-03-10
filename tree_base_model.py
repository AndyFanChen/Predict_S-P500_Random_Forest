import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.font_manager import FontProperties

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings

warnings.filterwarnings('ignore')


def creat_y(merged_data):
    merged_data['week_trend'] = np.where(
        merged_data['open'].shift(-2) > merged_data['open'].shift(-1), 1, 0)
    merged_data.dropna(inplace=True, axis=0)

    return merged_data

def data_split(merged_data):
    split_point = int(merged_data.shape[0] * 0.75)
    train = merged_data.iloc[:split_point, 1:].copy()
    test = merged_data.iloc[split_point:-1, 1:].copy()
    for i in range(train.shape[1]):
        train[train.columns[i]] = pd.to_numeric(train[train.columns[i]])
    for i in range(test.shape[1]):
        test[test.columns[i]] = pd.to_numeric(test[test.columns[i]])

    train_X = train.drop('week_trend', axis=1)
    train_X.dropna(inplace=True, axis=0)

    train_y = train.week_trend

    test_X = test.drop('week_trend', axis=1)
    test_X.dropna(inplace=True, axis=0)
    test_y = test.week_trend

    return train_X, train_y, test_X, test_y, test


def ml_training(train_X, train_y, test_X, test_y, max_features):
    # Instantiate model with 1000 decision trees
    # max_features = round(np.sqrt(merged_data.shape[1] - 1))

    # choose the model
    model = RandomForestClassifier(n_estimators=1500, oob_score=True, random_state=1,
                                max_features=max_features, max_depth=5, min_samples_leaf=3)
    # model = XGBClassifier()
    # model = LGBMClassifier()

    model.fit(train_X, train_y)
    oob_score = model.oob_score_
    print(f"oob_score {oob_score}")
    feature_names = train_X.columns
    
    # can choose feature by this score
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).sort_values()
    print(f"forest_importances {forest_importances}")

    prediction_training = model.predict(train_X)
    training_score = model.score(train_X, train_y)
    confusion_matrix_train = confusion_matrix(train_y, prediction_training)
    print(f"confusion_matrix_train {confusion_matrix_train}")
    prediction = model.predict(test_X)
    confusion_matrix_test = confusion_matrix(test_y, prediction)
    print(f"confusion_matrix_test {confusion_matrix_test}")

    return model, prediction, training_score

def prediction_to_data(merged_data, prediction):
    split_point = int(merged_data.shape[0] * 0.75)
    test = merged_data.iloc[split_point:-1, :].copy()
    test['prediction'] = prediction
    
    return test

def profit_cal(test):

    profit_list = []
    per_profit_list = []
    test.reset_index(drop=True, inplace=True)
    test.insert(test.shape[1], 'IfTrade', value="")
    test.insert(test.shape[1], 'profit', value="")
    cost = 1
    cost_count = 0
    for i in range(test.shape[0]):
        try:
            if test['prediction'].iloc[i] == 1:
                in_price = test['open'].iloc[i + 1]
                out_price = test['open'].iloc[i + 2]
                test['IfTrade'].iloc[i + 1] = "Long"
                profit = out_price - in_price
                if i == 0:
                    profit -= cost
                    cost_count += 1
                else:
                    if test['IfTrade'].iloc[i] != "Long":
                        profit -= cost
                        cost_count += 1
                test['profit'].iloc[i + 1] = profit
                profit_list.append(profit)
                per_profit = profit / in_price
                per_profit_list.append(per_profit)
            if i > 0:
                if test['prediction'].iloc[i] == 0 and test['week_trend'].iloc[i - 1] == 0:
                    in_price = test['open'].iloc[i + 1]
                    out_price = test['open'].iloc[i + 2]
                    test['IfTrade'].iloc[i + 1] = "Short"
                    profit = in_price - out_price
                    if i == 0:
                        profit -= cost
                        cost_count += 1
                    else:
                        if test['IfTrade'].iloc[i] != "Short":
                            profit -= cost
                            cost_count += 1
                    test['profit'].iloc[i + 1] = profit
                    profit_list.append(profit)
                    per_profit = profit / in_price
                    per_profit_list.append(per_profit)
            if test['IfTrade'].iloc[i] == "":
                test['profit'].iloc[i + 1] = 0
                per_profit_list.append(0)
        except IndexError:
            pass

    # np_profit = np.array(profit_list)
    # profit = np_profit.sum()
    # profit_std = np_profit.std()
    # np_per_profit = np.array(per_profit_list)
    # per_profit = np_per_profit.mean() * 252

    # buy and hold profit
    test['open損益'] = test['open'].shift(-1) - test['open']
    test.drop(test[test['profit']== ""].index, axis=0, inplace=True)
    SP500_profit = test['open損益'].to_numpy()
    sum_list_sp = []
    accumulate_sp = 0
    for i in range(SP500_profit.shape[0]):
        accumulate_sp += SP500_profit[i]
        sum_list_sp.append(accumulate_sp)
    # SP500_profit_sum = SP500_profit.sum()
    # SP500_profit_std = SP500_profit.std()
    # per_SP500_profit = ((test['open'].iloc[-1] / test['open'].iloc[0] - 1) / SP500_profit.shape[0]) * 252


    return profit_list, SP500_profit

def plot_loss(test, profit_list, SP500_profit, dpi=500):
    trade_date = test.loc[(test['IfTrade'] == "Long") | (test['IfTrade'] == "Short")]['交易日期']
    sum_list = []
    accumulate = 0
    for i in range(len(profit_list)):
        accumulate += profit_list[i]
        sum_list.append(accumulate)

    fig, ax = plt.subplots()
    font_set = FontProperties(fname=r"font_path.ttc", size=15, weight='bold')
    ax.plot(trade_date, sum_list)
    fig.autofmt_xdate()

    sum_list_sp = []
    accumulate_sp = 0
    for i in range(SP500_profit.shape[0]):
        accumulate_sp += SP500_profit[i]
        sum_list_sp.append(accumulate_sp)

    title = '策略與買進持有累積報酬絕對值比較'
    plt.title(title, fontproperties=font_set)
    ax.plot(test['交易日期'], sum_list_sp)
    font_set2 = FontProperties(fname=r"c:\windows\fonts\msjh.ttc", size=10, weight='bold')
    plt.xlabel('日期', fontproperties=font_set2)
    plt.ylabel('損益(點)', fontproperties=font_set2)
    font_set3 = FontProperties(fname=r"c:\windows\fonts\msjh.ttc", size=9, weight='bold')
    plt.legend(['機器學習策略報酬(點)', '買進持有報酬(點)'], prop=font_set3)
    fig.autofmt_xdate()
    plt.savefig('策略與買進持有累積報酬絕對值比較(隨機森林)', dpi=dpi)

    plt.show()

merged_data = pd.read_excel(r'merged_data_path.xlsx')
creat_y(merged_data)
train_X, train_y, test_X, test_y, test = data_split(merged_data)

max_features = round(np.sqrt(merged_data.shape[1] - 1))
prediction = ml_training(train_X, train_y, test_X, test_y, max_features)
test_data = prediction_to_data(merged_data, prediction)
profit_list, SP500_profit = profit_cal(test_data)
plot_loss(test_data, profit_list, SP500_profit, dpi=500)
