# Predict E-mini S&P 500 By Random Forest
* First step: preprocess and merge tech and macroeconomics data
* Second step: creat x and y then used random forest model to train use OOB score to do evaluation
* Third step: show the result about profit, volatility
## Execute
`python3 rf_sp500_predict.py`
## Preprocess and merge Data

Use `DataPreprocess` class to preprocess data

Creat technical indicator by historical price data, then merge all macroeconomics data might need, including index of main country (e.g. Nikkei 225, Hang Seng, ...), price of main goods(e.g. Gold, Copper, ...), and the other important data. 

## Execute the model and evaluate

Use `MLTrainEval` class to train and evaluate.

Define x (all data merged) and y (next day goes up or down), and split train set (2005 to 2017) and test set (2018 to 2021), use oob score to do validation, also select few most important indicator in this step.

## Result

<img src="https://hackmd.io/_uploads/H1ELiXynp.png" width="500" height="350">

X-axis: Date
Y-axis: Return (pips)
Dark green line: cumulative return (pips) of our strategy
Light Green line: cumulative return (pips) of buy and hold

<img src="https://hackmd.io/_uploads/HkCuR7knp.png" width="500" height="320">








