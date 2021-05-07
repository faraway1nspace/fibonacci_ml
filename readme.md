# Fibonacci ML: Automatic Fib Extensions/Retracements for Machine Learning

This repository contains code for automatically finding fibonacci-retracements in a price timeseries, and converting them _into features for statistical analysis_ (i.e., feature engineering).

The project attempts to remove the subjectiveness of drawing fibonacci retracements, and then, having a drawdown + fib-retracement, a means to convert the fib-levels into a workable timeseries that can be ingested for time-series analyses of price.

- Inputs:  
-- pandas dataframe of OHLC prices
- Outputs:  
-- pandas dataframe of features corresponding to a contiguous time-series representing *all* fib-retracements and extensions.

The main features and benefits of the approach are the following:
- automatically finds drawdowns with either: a user-specified `drawdown_criteria` (usually 0.2) or finds an appropriate criteria automatically
- converts the fibs into a smooth contiguous timeseries of features of the drawdown (`max_drawdown`,`duration`, `precovery` (percent recovery), `fib_lev` (the fib-level), `time_since_peak_d`, and `fib-box01` (the % that price is between two fib-levels).
- tracks three fibonacci-retracements in parallel: the current/most recent; the previous; and the "long-term memory" of any significant monster drawdowns (even decades earlier)
- doesn't "cheap": the features at time `t` are never calculated using information from the future, which many backtesters violate when they first make fib-retracements for an entire time-series, and then fail to mask/hide the future information from prices in the past (illegal!)

The figure below (top) shows the automatic finding of retracements and extensions for `QQQ`. Notice that the levels correspond to: `0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618, 4.236, 6.854, 11.09, 17.944`

![](img/fibonacci_timeseries.png?raw=true)


Summary of features for `QQQ`.

```                          count      mean       std       min       25%       50%       75%       max
fib-1_max_drawdown_d     5578.0  0.440728  0.150744  0.204047  0.282772  0.530009  0.530974  0.767416
fib-1_time_since_peak_d  5578.0  6.541665  0.969640  2.772589  5.937536  6.781058  7.321189  7.953318
fib-1_duration_d         5578.0  6.031123  1.106973  2.772589  5.003946  6.389924  6.997596  7.254885
fib-1_precovery_d        5578.0 -0.025882  0.352544 -1.090901 -0.201193  0.002114  0.178860  0.767416
fib-1_fib_lev_d          5578.0 -0.114710  0.542956 -1.000000 -0.432416 -0.055316  0.098612  1.248340
fib-1_box01_d            5578.0  0.458309  0.280091  0.000000  0.207130  0.451929  0.689416  0.999762
fib-2_max_drawdown_d     5578.0  0.580211  0.178354  0.204047  0.530009  0.530974  0.799657  0.830577
fib-2_time_since_peak_d  5578.0  7.759018  0.507455  6.265301  7.523076  7.824046  8.162516  8.511779
fib-2_duration_d         5578.0  6.936482  0.810381  4.369448  6.837995  7.254885  7.254885  7.956126
fib-2_precovery_d        5578.0 -0.346507  0.803978 -2.420680 -0.814035 -0.522701  0.585802  0.830577
fib-2_fib_lev_d          5578.0 -0.019981  0.838561 -1.000000 -1.000000  0.098612  0.830339  1.248340
fib-2_box01_d            5578.0  0.466216  0.273921  0.001096  0.245649  0.448293  0.700172  0.999992
fib-3_max_drawdown_d     5578.0  0.759640  0.086926  0.653140  0.653140  0.830577  0.830577  0.830577
fib-3_time_since_peak_d  5578.0  8.444213  0.243571  7.957527  8.308650  8.308650  8.645894  8.950403
fib-3_duration_d         5578.0  8.099524  0.506587  7.518228  7.518228  8.206993  8.645894  8.694670
fib-3_precovery_d        5578.0 -0.255281  0.599137 -1.905022 -0.711667 -0.526383  0.358030  0.782663
fib-3_fib_lev_d          5578.0 -0.503271  0.543082 -1.000000 -1.000000 -0.613378 -0.055316  0.830339
fib-3_box01_d            5578.0  0.471312  0.235317  0.000014  0.388402  0.459239  0.545765  0.999927
```

## Demonstration

See the file `demo_fibonacci_ml.py`.

```
import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import re

from fibonacci_ml.core import *

# get timeseries for QQQ
data = yf.download("QQQ")

# initialize the FibonacciTechnicalAnalysis object
fib_maker = FibonacciTechnicalAnalysis(data, drawdown_criteria=0.20, do_plot=False)

# make the features
features = fib_maker.make_fib_features()
```

The pandas object `features` can then be used with price or other TA features in a machine-learning model for time-series analysis. That is what I do!

## Price Snaking Through Fib-Levels

Here is the QQQ price as it snakes through various fibonacci-retracement and extension levels.

![](img/fib-snake.png?raw=true)


## Memory

The `FibonacciTechnicalAnalysis` has 3 memories: it tracks three retracements in parallel so that price is "aware" of multiple drawdowns
- Memory 1: the current/most recent drawdown. This is typically what most analysts focus on for short-term pivots  
- Memory 2: the previous drawdown. TA analysts often pay attention when the fib-levels from two different drawdowns align.  
- Long-term Memory: a model is used to track the long-term monster drawdowns, often spanning decades. E.g., some TA analysts refer to the 1999 Nasdaq drawdown for fib-extensions.

The following two graphs compare the Memory-1 drawdowns vs the Long-Term Memory drawdowns. The blue-dash lines represent, at any given point in time, which fib-levels are in the current "memory" and thus exposed to price. Notice that Memory-1, the fib levels are changing with each new drawdown. However, the long-term memory seems to *only* can about the monster-drawdown in 1999.

### short-term memory:

![](img/shortterm-memory.png?raw=true)

### long-term memory:

![](img/longterm-memory.png?raw=true)

