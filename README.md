# Predict E-mini S&P 500 By Random Forest

## Result
* achieves an annualized return 9% higher than the market index, with lower volatility than the market index  
<img src="https://github.com/AndyFanChen/SP500_Random_Forest_Predict/blob/main/Profit_Plot.png" width="500" height="350">

* X-axis: Date  
* Y-axis: Return (pips)  
* Dark green line: cumulative return (pips) of our strategy  
* Light Green line: cumulative return (pips) of buy and hold

<img src="https://github.com/AndyFanChen/SP500_Random_Forest_Predict/blob/main/Return_Table.png" width="500" height="320">

## Preprocess and merge Data

Use `data_preprocess.py` to preprocess data

Creat technical indicator by historical price data, then merge all macroeconomics data might need, including index of main country (e.g. Nikkei 225, Hang Seng, ...), price of main goods(e.g. Gold, Copper, ...), and the other important data. 


## Start Training
`python3 tree_base_model.py`

* First step: preprocess and merge tech and macroeconomics data
* Second step: creat x and y then used random forest model(also can use other tree base model like XGBoost LGBM) to train use OOB score to do evaluation
* Third step: show the result about profit, volatility

Define x (all data merged) and y (next day goes up or down), and split train set (2005 to 2017) and test set (2018 to 2021), use oob score to do validation, also select few most important indicator in this step.










