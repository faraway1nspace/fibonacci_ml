# Fibonacci ML: Automatic Fib Extensions/Retracements for Machine Learning

This repository contains code for automatically finding fibonacci-retracements in a price timeseries, and converting them _into features for statistical analysis_ (i.e., feature engineering).

The project attempts to remove the subjectiveness of drawing fibonacci retracements, and then, having a drawdown + fib-retracement, a means to convert the fib-levels into a workable timeseries that can be ingested for time-series analyses of price.

- Inputs:
-- pandas dataframe of OHLC prices
- Outputs:
-- pandas dataframe of features corresponding to a contiguous time-series representing *all* fib-retracements and extensions.

![](img/fibonacci_timeseries.png?raw=true)