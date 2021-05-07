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

