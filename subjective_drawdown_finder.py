# classes and functions to automatically determine a good drawdown-criteria (% drawdown peak to tough) for making fibonacci-extensions/retractements
# uses two boosted-trees models, plus some heuristics, in order to find a good criteria
import os
import datetime
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import copy
import re
import pickle
from sklearn.tree import DecisionTreeRegressor

from .variables import RECOVERY_CRITERIA, TARGET_DENSITY
from .fib_utils import *

CORE_PATH = os.path.abspath(os.path.dirname(__file__))

# main class of this file
class SubjectiveDrawdown:
    """
    models and functios to find optimal drawdown for making fibonacci extensions 
    principal function is self.fit()
    """
    def __init__(self, verbose =None, target_density=None, drawdown_cap = None, recovery_criteria=None, path_to_model_pred = None, path_to_model_refine = None):
        
        if verbose is None:
            verbose = False
        self.verbose = verbose
        
        # default target_density
        if target_density is None:
            target_density=TARGET_DENSITY
        self.target_density = target_density
        
        # cap the range of plausible drawdown criteria
        if drawdown_cap is None:
            drawdown_cap = [0.05, 0.7]
        self.drawdown_cap= drawdown_cap
        
        # criteria to judge when a retracement is finished (from peak)
        if recovery_criteria is None:
            recovery_criteria = RECOVERY_CRITERIA
        self.recovery_criteria = recovery_criteria
        
        # load the probabilistic models
        self.model = SubjectiveDrawdownModels(path_to_model_pred = path_to_model_pred,
                                              path_to_model_refine = path_to_model_refine,
                                              verbose=verbose)            
    
    def prefeature_trend(self, data, focal_column=None):
        """ mean and std (around residuals)"""
        if focal_column is None:
            focal_column = 'Close'
        
        # y data
        y = np.log(np.clip(data[focal_column], a_min = 0.001, a_max = None))
        y = ((y-y.mean()).values)#/y.std()
        
        # x data
        x = ((data.index - data.index.mean()).days).values/365
        
        # slope and intercept
        m = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) * np.sum(x)) # long-run log-linear increase
        b = (np.sum(y) - m *np.sum(x)) / len(x)
        
        # take the variance around the dominant trend
        residuals = y-(x*m+b)
        std_ = residuals.std()
        
        return m,std_
    
    def prefreature_realizedvol(self, data, hlc_columns=['High', 'Low', 'Close']):
        """basically standard-deviation, notice we exclude open because of API issues
        instead of std from the mean price, we take it from the previous price
        """
        
        # log the prices for highh low close
        y_hlc = [np.log(np.clip(data[col].values,a_min=0.01,a_max=None)) for col in hlc_columns]
        
        # split into hi-close and lo-close
        y_hc = np.concatenate([y_hlc[0].reshape(-1,1)]+[y_hlc[-1].reshape(-1,1)], axis=1).reshape(-1)
        y_lc = np.concatenate([y_hlc[1].reshape(-1,1)]+[y_hlc[-1].reshape(-1,1)], axis=1).reshape(-1)
        
        # difference between close and previous hi
        mean_realized_volatility = (((np.diff(y_hc)**2).sum() + (np.diff(y_lc)**2).sum())/(len(y_hc) + len(y_lc)-2))**0.5
        
        # same as above, but limited to only downsides
        y_close_diff = np.diff(y_hlc[-1])
        y_close_downside_diff = y_close_diff[np.where(y_close_diff<=0)[0]]
        mean_downside_volatility = ((y_close_downside_diff**2).mean())**0.5
        
        #
        return mean_realized_volatility, mean_downside_volatility
    
    def _optimal_drawdown_for_fibs_probablistic_estimator(self, data, target_density):
        """estimate an initial drawdown criteria, through a probabilistic model"""
        # get features: trend and std
        ftrend,fstd = self.prefeature_trend(data)
        if (str(ftrend)=='nan') or str(fstd)=='nan':
            raise ValueError("trend or std")
        
        # get features: volatility and downside vol
        fvol, fvoldown = self.prefreature_realizedvol(data, hlc_columns=['High', 'Low', 'Close'])
        if (str(fvol)=='nan') or str(fvoldown)=='nan':
            raise ValueError("trend or std")
        
        # features must be ordered: ['drawdown_crit', 'trend', 'std', 'vol', 'vold']
        drawdown_criterias = np.linspace(0.05, 0.5, 50).reshape(-1,1)
        X = np.concatenate([drawdown_criterias, np.array([ftrend]*50).reshape(-1,1), np.array([fstd]*50).reshape(-1,1), np.array([fvol]*50).reshape(-1,1), np.array([fvoldown]*50).reshape(-1,1)],axis=1)
        # pdensity
        pdensity = self.model.predict(X)
        
        # drawdown criteria suggested
        drawdown_crit_suggested = drawdown_criterias[np.argmin((pdensity - target_density)**2)][0]
        
        return drawdown_crit_suggested, [ftrend, fstd, fvol, fvoldown, target_density]
    
    def _get_fibs(self, data, drawdown_crit, recovery_criteria=None):
        """ wrapper for find_all_retracement_boxes and Fib to make a time-series of fibs"""
        if recovery_criteria is None:
            recovery_criteria = self.recovery_criteria
        
        fib_spans = find_all_retracement_boxes(data, drawdown_criteria=drawdown_crit)
        # Fib(fib_span=fib_span, data=self.data, drawdown_criteria=self.drawdown_criteria, fib_levels=sexlf.fib_levels, recovery_criteria = self.recovery_criteria, make_features = make_features)
        fib_series = [Fib(fib_span=fib_span, data=data, drawdown_criteria=drawdown_crit, recovery_criteria=0.02, fib_levels = [0,1,1.618]) for fib_span in fib_spans]
        # remove null fibs (must pass .is_fib)
        fib_series = [fib for fib in fib_series if fib.is_fib]
        return fib_series
    
    def _density_of_drawdowns_given_fibs(self, fib_series, data=None, delta_time=None):
        """estimates the annual density of fibs"""
        # total time duration of series
        if delta_time is None:
            delta_time = (data.index[-1] - data.index[0]).days/365
        return len(fib_series)/delta_time
    
    def _densities_by_kulling(self, fib_series, delta_time, orig_drawdown = None, results = None):
        """empirical calculation of the relationship between drawdown and densities, by progressivingly kulling drawdowns"""
        # maxdrawdowns
        if orig_drawdown is None:
            orig_drawdown = 0.2
        
        # results
        if results is None:
            results = pd.DataFrame({'drawdown_crit':[orig_drawdown], 'density':[len(fib_series)/delta_time]})
        if len(fib_series)==0:
            return results
        
        max_drawdowns = [fib.features['max_drawdown'].max() for fib in fib_series]
        max_drawdowns = sorted(max_drawdowns)
        
        for i,drawdown_crit in enumerate(max_drawdowns):
            
            density_ =[len(max_drawdowns[(i+1):])/delta_time]
            
            crit_ = [drawdown_crit*1.001]
            
            if (np.abs(drawdown_crit*1.001 - results['drawdown_crit'].values).min() > 0.0005):
                results = results.append(pd.DataFrame({'drawdown_crit':crit_, 'density':density_}))
        
        return results
    
    def _drawdown_manual_finder(self, data, results, target_density, increment = None):
        """
        uses recursion to find a target density
        increments a drawdown by 'increment' multiplicatively
        """
        # do recursion if all results are 0, or no results are greater than target
        do_recursion = (results['density']==0).all() or (not (results['density'] >= target_density).any() )
        
        if not do_recursion:
            return results
        
        if increment is None:
            increment = 0.95
        
        drawdown_crit_increment = increment*results['drawdown_crit'].min()
        
        fib_series = self._get_fibs(data, drawdown_crit_increment)
        delta_time = (data.index[-1] - data.index[0]).days/365
        
        # initial results
        results = results.append(pd.DataFrame({'drawdown_crit':[drawdown_crit_increment],
                                           'density':[1.001*len(fib_series)/delta_time]}))
        # initial empirical results
        results = self._densities_by_kulling(fib_series,
                                            delta_time,
                                            orig_drawdown =drawdown_crit_increment,
                                            results = results)
        
        do_recursion = (results['density']==0).all() or (not (results['density'] >= target_density).any())
        if do_recursion:
            return self._drawdown_manual_finder(data, results, target_density, increment)
        
        return results
    
    def fit(self, data, target_density=None, drawdown_cap=None):
        """estimate an initial drawdown criteria, through:
        - step1: a probabilistic model
        - step2: iterate through fibs and kull one-by-one, empirically measuring the densities"""
        assert type(data) == pd.core.frame.DataFrame
        assert 'Close' in data.columns
        
        if target_density is None:
            target_density = self.target_density
        
        if drawdown_cap is None:
            drawdown_cap = self.drawdown_cap
        
        drawdown_crit, X = self._optimal_drawdown_for_fibs_probablistic_estimator(data, target_density)
        # get fibs and calculate the density of drawdownd
        fibs = self._get_fibs(data, drawdown_crit)
        
        # calculate density and residuals
        delta_time = (data.index[-1] - data.index[0]).days/365
        realized_density = 1.001*len(fibs)/delta_time
        resid = target_density - realized_density
        
        if self.verbose:
            print("%s: DD1 %0.3f:%0.3f fibs/year" % (ticker, drawdown_crit, realized_density))
        
        # initial results
        results = pd.DataFrame({'drawdown_crit':[drawdown_crit], 'density':[realized_density]})
        
        # next estimate: trigger next model
        if realized_density < target_density:
            
            # run next model (refinement)
            X += [drawdown_crit, resid]
            drawdown_crit = self.model.refine(X)#[0]
            fibs = self._get_fibs(data, drawdown_crit)
            
            realized_density = 1.001*len(fibs)/delta_time
            results = pd.DataFrame({'drawdown_crit':[drawdown_crit], 'density':[realized_density]})
            if self.verbose:
                print("%s: DD2 %0.3f:%0.3f fibs/year" % (ticker, drawdown_crit, realized_density))
        
        # initial empirical results
        results = self._densities_by_kulling(fibs, delta_time, orig_drawdown =drawdown_crit, results = results)
        # recursively find drawdown closer to the target
        results = self._drawdown_manual_finder(data, results, target_density)
        
        # get what?: at least as great as the target_density, but closeest
        ix_meet_or_exceed_criteria = np.where(results.density >= target_density)[0]
        
        if len(ix_meet_or_exceed_criteria)>0:
            results_sub = results.iloc[ix_meet_or_exceed_criteria]
        else:
            results_sub = results
        
        drawdown_crit_suggested = results_sub.drawdown_crit.iat[np.argmin((results_sub.density - target_density)**2)]
        if self.verbose:
            print("%s: DD3 %0.3f FINAL" % (ticker, drawdown_crit_suggested))
        
        # clip the drawdown output
        if drawdown_crit_suggested> max(drawdown_cap):
            drawdown_crit_suggested = max(drawdown_cap)
        elif drawdown_crit_suggested < min(drawdown_cap):
            drawdown_crit_suggested = min(drawdown_cap)
        
        return drawdown_crit_suggested, results

class SubjectiveDrawdownModels:
    """container for two boosting models that predict drawdown-criterias"""
    def __init__(self, path_to_model_pred = None, path_to_model_refine = None, verbose=False, unit_test = True):
        self.verbose = verbose
        #print("current_path; %s" % current_path)
        if path_to_model_pred is None:
            path_to_model_pred = os.path.join(CORE_PATH, "subjective_drawdown_models/subjective_drawdown_model1.pkl")
        if path_to_model_refine is None:
            path_to_model_refine = os.path.join(CORE_PATH, "subjective_drawdown_models/subjective_drawdown_model2.pkl")
        
        self.path_to_model_pred = path_to_model_pred
        self.path_to_model_refine = path_to_model_refine
        
        # load the models
        self.load_model_pred()
        self.load_model_refine()
        
        # unit_test on load
        if unit_test:
            self.run_tests()
    
    def load_model_pred(self):
        """load the model one/predictor model (sklearn boosted regression trees) """
        if self.verbose:
            print("loading drawdown prediction model 1 %s" % self.path_to_model_pred)
        with open(self.path_to_model_pred, 'rb') as pcon:
            mod_pred = pickle.load(pcon)
        self.mod_pred = mod_pred
    
    def load_model_refine(self):
        """load the model two/refiner model (sklearn boosted regression trees) """
        if self.verbose:
            print("loading drawdown refinement model 2 %s" % self.path_to_model_refine)
        with open(self.path_to_model_refine, 'rb') as pcon:
            mod_refine = pickle.load(pcon)
        self.mod_refine = mod_refine
    
    def predict(self, X):
        """prediction from model one"""
        if isinstance(X, list):
            X = np.array(X).reshape(1,-1)
        
        return self.mod_pred.predict(X)
    
    def refine(self, X2):
        """conditional one the residuals from model one, and one data-download, """
        if isinstance(X2, list):
            X2 = np.array(X2).reshape(1,-1)
        return self.mod_refine.predict(X2)[0]
    
    def run_tests(self):
        """ units tests on models"""
        p = self.predict(np.array([[0.58720078, 0.39931927, 0.2731371 , 0.04188929, 0.0312278 ]]))
        print("testing model 1 (predictor)")   
        assert (p[0] - 0.28313829505556226) < 10**-6
        
        q = self.refine(np.array([[0.39931927012611657, 0.273137095058999, 0.041889285755502804, 0.03122779683661232, 0.26, 0.4908163265306123, -4.626334519569619e-05]]))
        print("testing model 2 (refiner)")
        assert (q - 0.46655745847591307) < 10**-6

#foo = SubjectiveDrawdownModels(unit_test = True)
#subjective_drawdown = SubjectiveDrawdown(verbose =True, target_density=0.25)

