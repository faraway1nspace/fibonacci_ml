import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
from .variables import TARGET_DENSITY, RECOVERY_CRITERIA, FIB_LEVELS

# Fib, find_all_retracement_boxes, find_retracement_boxes, get_highs, get_lows

def get_highs(data):
    """takes average  high and max(close,open). I.e., split the difference between a body-candle and a wick"""
    return (0.5*data['High'] +0.5*data[['Close','Open']].max(axis=1))

def get_lows(data):
    """takes average of low and low(close,open)"""    
    return (0.5*data['Low'] +0.5*data[['Close','Open']].min(axis=1))

def find_retracement_boxes(data, drawdown_criteria,do_plot=False, offset=None, recovery_criteria=None):
    """when drawing fibonacci retrace/extensions, you need to draw a box from the local high to low; this function automatically finds such boxes on which to base the retracement/extension-levels;
    returns list of tuples [(m1,2),...]. m1:= start of retracement box; m2:=end of retrace box"""
    if offset is None:
        offset = 0
    
    if recovery_criteria is None:
        recovery_criteria = 0.02
    
    # cumulative highs during period
    vHighs = get_highs(data)
    cummax = vHighs.cummax()
    # vector of % draw-dwosn
    vDrawdowns = (cummax - get_lows(data))/cummax
    vRecovery = (cummax - vHighs)/cummax
    
    # crude binary indicator of whether or not price is in a drawdown
    #in_Drawdowns = vDrawdowns>=drawdown_criteria
    
    rDrawdowns = []
    r_drawdown = 0
    #in_drawdown = 0
    # loop backwards
    for i in range(len(vDrawdowns)-1,-1,-1):
        #in_drawdown_lag = in_drawdown
        #in_drawdown = in_Drawdowns.iloc[i]
        #reset_ = (vDrawdowns.iloc[i]<=recovery_criteria)
        reset_ = (vRecovery.iloc[i]<=recovery_criteria)
        r_drawdown = max(r_drawdown, vDrawdowns.iloc[i]) if not reset_ else 0
        rDrawdowns.append(r_drawdown)
    
    rDrawdowns = np.array(rDrawdowns[::-1])
    
    # putative fib periods (vector
    fib_periods = rDrawdowns*(rDrawdowns>=drawdown_criteria) + np.zeros(len(rDrawdowns))
    
    # fib periods are useful for finding local fib periods
    # now split the fib periods into spans
    idx_fib_periods = np.where(np.diff(1*(fib_periods>0)))[0]
    if fib_periods[-1]>0:
        idx_fib_periods = np.concatenate((idx_fib_periods, np.array([len(fib_periods)])))
    
    nmax = data.shape[0] 
    #fib_spans = [(m1+1,min(m2+1,nmax )) for m1,m2 in zip(idx_fib_periods[:-1],idx_fib_periods[1:]) if all(fib_periods[(m1+1):(m2+1)]>0)]
    fib_spans = [(m1,min(m2+1,nmax )) for m1,m2 in zip(idx_fib_periods[:-1],idx_fib_periods[1:]) if all(fib_periods[(m1+1):(m2+1)]>0)]
    
    if do_plot:
        fig, axs = plt.subplots(3)
        axs[0].plot(np.arange(data.shape[0]),np.log(data['Close']))    
        axs[1].plot(np.arange(vDrawdowns.shape[0]),vDrawdowns)
        # plotting the cumsum
        axs[2].plot(np.arange(len(rDrawdowns)),rDrawdowns[::-1])
        for s in fib_spans:
            axs[2].plot(np.arange(s[0],s[1]), [0.5]*(s[1]-s[0]))
        plt.show()
    
    if offset!=0:
        # adjust the spans by offset
        fib_spans = [(m1+offset,m2+offset) for m1,m2 in fib_spans]
    return fib_spans

# finding smmaler-fib periods WITHIN giant fib periods
def find_sub_drawdowns_within_a_giant_drawdown(data, fib_span, drawdown_criteria, START_LOOKING_AFTER_DAYS=None, POST_BUFFER_DAYS = None, RETRACE_MINIMUM=None, recovery_criteria=None):
    """
    finds drawsdowns within larger drawdowns, using some criteria:
    - time: only starts looking for another drawdown after 1.5 years after the peak of the supra
    - minimum retrace: the supra must retrace to 0.5 level to qualify for looking for another drawdown (to prevent too many drawdowns that are just continuation of the primary trend
    """
    if START_LOOKING_AFTER_DAYS is None:
        START_LOOKING_AFTER_DAYS = 252*1.5
    if POST_BUFFER_DAYS is None:
        POST_BUFFER_DAYS = 100
    if RETRACE_MINIMUM is None:
        RETRACE_MINIMUM=0.382
    if recovery_criteria is None:
        recovery_criteria=0.02
    
    start_iloc, stop_iloc = fib_span
    n_ = data.shape[0]
    if ((stop_iloc-start_iloc)<=START_LOOKING_AFTER_DAYS) or ((start_iloc+START_LOOKING_AFTER_DAYS)>= n_):
        # don't proceed if small retracement period
        return None
    
    # get the retracements: check if it has retrace at least X
    subdata = data.iloc[start_iloc:stop_iloc]
    cummax_ = get_highs(subdata).cummax()
    low_ = get_lows(subdata)
    cummin_ = low_.cummin()
    vDrawdowns_ = (cummax_ - low_)/cummax_
    vDoes_retracement = subdata['Close']>=(((cummax_ - cummin_)*RETRACE_MINIMUM)+cummin_)
    if not vDoes_retracement.sum():
        # no price is above the minium retracement
        return None
    
    # minimum place to start relooking for another drawdown
    ix_minrestart = vDoes_retracement.tolist().index(1)
    ix_minrestart = max(ix_minrestart, START_LOOKING_AFTER_DAYS)
    
    # new (crude) search box
    startsub_iloc = int(start_iloc + min(ix_minrestart,n_) )
    stopsub_iloc = int(stop_iloc + min(POST_BUFFER_DAYS, n_))
    fib_subspans = find_retracement_boxes(data.iloc[startsub_iloc:stopsub_iloc], drawdown_criteria, do_plot=False, offset = startsub_iloc, recovery_criteria = recovery_criteria)
    return fib_subspans

# wrapper for find_retracement_boxes
def find_all_retracement_boxes(data, drawdown_criteria=None, START_LOOKING_AFTER_DAYS=None, POST_BUFFER_DAYS = None, fib_spans=None, recovery_criteria=None):
    """
    combines 'find_retracement_boxes' and 'find_sub_drawdowns_within_a_giant_drawdown'
    used recursively
    """
    # big spans
    if fib_spans is None:
        fib_spans_big = find_retracement_boxes(data, drawdown_criteria,do_plot=False, recovery_criteria=recovery_criteria)
    else:
        fib_spans_big = fib_spans
    
    # find smaller subspans within big spans
    fib_spans = [] # container
    for i, fib_span in enumerate(fib_spans_big):
        
        fib_spans.append(fib_span)
        
        # find subsspan within fib_span
        fib_subspans = find_sub_drawdowns_within_a_giant_drawdown(data, fib_span, drawdown_criteria, START_LOOKING_AFTER_DAYS, POST_BUFFER_DAYS, recovery_criteria=recovery_criteria)
        # integrate
        if not (fib_subspans is None):
            fib_subspans = find_all_retracement_boxes(data, drawdown_criteria, START_LOOKING_AFTER_DAYS, POST_BUFFER_DAYS, fib_subspans, recovery_criteria = recovery_criteria)
            for subspan in fib_subspans:
                if subspan not in fib_spans:
                    fib_spans.append(subspan)
    
    return fib_spans

class Fib:
    """contains necessary data to make a fibonacci retracement into a feature for ML"""
    def __init__(self, fib_span, data, drawdown_criteria, fib_levels, recovery_criteria, make_features = True):
        self.drawdown_criteria = drawdown_criteria # what is considered a bear market crash?
        self.start = fib_span[0] # start of retracement
        self.end = fib_span[1]   # end (no longer 20% drawdown)
        self.fib_levels = np.array(fib_levels) # fib numbers (0.
        self.recovery_criteria = recovery_criteria # percent to high that declares bear over
        
        # default: not a fib 
        self.is_fib = False
        self.indx_start_of_credible_fib = None
        self.loc_start_of_credible_fib = None
        #self.loc_end = data.index[self.end] # how to use this????? because the actualy index is -1
        
        # get levels, as a numpy time-series
        fib_series, series_indices = self.calc_fib_series_on_span(data, fib_span, do_mask=True)
        self.is_fib = not (fib_series is None)
        
        # features for ML tool        
        if make_features and self.is_fib:
            features = self._make_features(data)

        #
        self.n_total = data.shape[0]
    
    def calc_fibs(self, hi,lo):
        """given a high, and a low, get the fibinocci extensions and retracements. """
        return (hi-lo)*self.fib_levels + lo
    
    def calc_fib_series_on_span(self, data, fib_span=None, do_mask=True, drawdown_criteria=None):
        """given a price series, and two indices that box-in the draw-down, it makes fibonnaci retracements"""
        if drawdown_criteria is None:
            drawdown_criteria = self.drawdown_criteria
        
        if fib_span is None:
            start_iloc, stop_iloc = self.start, self.end
        else:
            start_iloc, stop_iloc = fib_span
        
        subdata = data.iloc[start_iloc:stop_iloc]
        # get cummulative-high (notice it takes halfway between body-of-candle and wick
        cummax = get_highs(subdata).cummax()
        # get cummulative low
        cumlow = get_lows(subdata).cummin()
        # series of fibs
        fib_series_ = FibTimeSeries(cumlow, cummax, fib_levels = self.fib_levels, indices_extended = data.index[data.index>=subdata.index[0]])
        indices = np.arange(start_iloc, stop_iloc)
        
        # mask
        if do_mask:
            # mask out all fibs BEFORE the 20% drawdown (because at those times, we wouldn't know we would soon be making fib-retracements
            if not (self.indx_start_of_credible_fib is None):
                indx_start_of_credible_fib = self.indx_start_of_credible_fib
            else:
                in_drawdown = 1*(((cummax - cumlow)/cummax)>=self.drawdown_criteria)
                if not in_drawdown.any():
                    return None,[]
                
                indx_start_of_credible_fib = in_drawdown.tolist().index(1)
                self.indx_start_of_credible_fib = indx_start_of_credible_fib
                self.loc_start_of_credible_fib = data.index[self.indx_start_of_credible_fib]
            
            # truncate series
            fib_series_.mask_out_predrawdown(indx_start_of_credible_fib)
            # new indices (after truncating for the first drawdown
            indices = indices[indx_start_of_credible_fib:]
            assert len(indices) == fib_series_.shape[-1]
        
        self.fib_series = fib_series_
        self.series_indices = indices
        return fib_series_, indices
    
    def _make_features(self, data, fib_span=None, drawdown_criteria=None, recovery_criteria=None):
        """ makes primatives for calculating fib retracements and features like:
        - max drawdown
        - duration
        - ever-recovers?
        Returns data as a dict
        Returans two versions of the data: 
        i) in_span: the values valid within the drawdown phase
        ii) extended: values extended beyond the span, to the end of the (global) time-series
        """
        if recovery_criteria is None:
            recovery_criteria = self.recovery_criteria
        
        if drawdown_criteria is None:
            drawdown_criteria = self.drawdown_criteria
        
        if fib_span is None:
            fib_span = (self.start,self.end)
        
        # beginning and end of drawdown
        start_iloc, stop_iloc = fib_span
        
        subdata = data.iloc[start_iloc:stop_iloc]
        
        # size of span
        n = stop_iloc-start_iloc
        n_extended = data.shape[0]-stop_iloc
        
        # get cummulative-high (notice it takes halfway between body-of-candle and wick
        cummax = get_highs(subdata).cummax()
        # get last price-value at time of recovery
        final_price_at_recovery = cummax[-1]
        # cummax extended to end of dataseries
        cummax_extended = pd.DataFrame({'cummax':[final_price_at_recovery]*n_extended}, index = data.index[stop_iloc:])['cummax']
        # cummulative lows (extended and in_span)
        low_extended = get_lows(data.iloc[start_iloc:]) # extended
        low_ = low_extended.iloc[:n]
        cumlow = low_.cummin()
        
        # r_drawdown: proportion drawdown
        vDrawdown = (cummax - low_)/cummax
        vDrawdown_extended = (cummax_extended - low_extended[n:])/cummax_extended
        vDrawdown_full = pd.concat((vDrawdown,vDrawdown_extended),axis=0)
        
        # is in drawdown?
        in_drawdown = 1*(((cummax - cumlow)/cummax)>=drawdown_criteria)
        if not in_drawdown.any():
            return None
        
        # mask: in realtime, we only know we are in a fib if drawdown criteria is met
        self.indx_start_of_credible_fib = in_drawdown.values.argmax()
        self.loc_start_of_credible_fib = in_drawdown.index[self.indx_start_of_credible_fib]
                
        # feature: has recovered?
        feat_recovered_full = 1*(vDrawdown_full[self.indx_start_of_credible_fib:]<recovery_criteria).cummax()
        feat_recovered = feat_recovered_full.iloc[:(n - self.indx_start_of_credible_fib)]
        feat_recovered_extend = feat_recovered_full.iloc[(n - self.indx_start_of_credible_fib):]
        # Done feature: has recovered
        
        # get the date-at-recovery
        self.time_at_recovery = (feat_recovered_full==1).idxmax(axis=0) # when first recovered?
        if self.time_at_recovery == feat_recovered_full.index[0]:
            self.time_at_recovery = feat_recovered_full.index[-1] # set t_at_recovery to enddate
        self.idx_at_recovery = np.where(data.index ==self.time_at_recovery)[0][0]
        
        # feature: time size peak
        # ... get index of the peak (during the fib_span) (
        self.loc_peak = cummax.idxmax() # index of peak
        vTimeSincePeak = pd.DataFrame({'vTimeSincePeak':((subdata.index - self.loc_peak).days).values},index=subdata.index)['vTimeSincePeak']
        vTimeSincePeak_extended = pd.DataFrame({'vTimeSincePeak':((data.iloc[fib_span[1]:].index - self.loc_peak).days).values},index = data.index[fib_span[1]:])['vTimeSincePeak']
        vTimeSincePeak_full = pd.concat((vTimeSincePeak,vTimeSincePeak_extended),axis=0).iloc[self.indx_start_of_credible_fib:] 
        # Done feature: time since peak
        
        # duration: how long was the drawdown (fixed)
        duration_full = pd.concat((
            pd.DataFrame({'duration':vTimeSincePeak_full.loc[:(self.time_at_recovery+pd.Timedelta(-1, unit="day"))]})['duration'],
            pd.DataFrame({'duration':[vTimeSincePeak_full.loc[self.time_at_recovery]]*(data.shape[0]-self.idx_at_recovery)}, index = data.loc[self.time_at_recovery:].index)['duration']
            ), axis=0)
        
        # feature: max drawdown
        max_drawdown = ((cummax-cumlow)/cummax)
        max_drawdown_extended = pd.DataFrame({'max_drawdown':[max_drawdown.max()]*n_extended},index=data.index[stop_iloc:])['max_drawdown']
        max_drawdown_full = pd.concat((max_drawdown,max_drawdown_extended),axis=0).iloc[self.indx_start_of_credible_fib:] 
        
        # feature volume: cumsum(price below peak) summed from peak to end of recovery
        # formula: ("is_recovered?") x (drawdown percentage) vDrawdown)
        # ... the cumsum is the naive volume. To get actual volume, we need to multiply by the number of days inbetween points, then cumsum
        volume_naive_pdf = (1-feat_recovered_full)*vDrawdown_full.iloc[self.indx_start_of_credible_fib:]
        # ... do ('volume_naive_pdf' x diff(days)).cumsum() to get absolute volume
        diffs_days = (volume_naive_pdf.index[1:] - volume_naive_pdf.index[:-1]).days.values
        diffs_days = np.array([diffs_days.mean()] + diffs_days.tolist()) # 
        volume = (volume_naive_pdf*diffs_days).cumsum()
        
        # collect features
        self.features = {'max_drawdown':max_drawdown_full,
                        'time_since_peak':vTimeSincePeak_full,
                        'duration':duration_full,
                        'recovered':feat_recovered_full,
                        'precovery':vDrawdown_full.iloc[self.indx_start_of_credible_fib:],
                         'volume': volume
                        }
        
        assert self.features['max_drawdown'].shape[0] ==self.features['time_since_peak'].shape[0]
        assert self.features['recovered'].shape[0] ==self.features['time_since_peak'].shape[0]
        assert self.features['precovery'].shape[0] ==self.features['time_since_peak'].shape[0]
        assert self.features['precovery'].shape[0] ==self.features['volume'].shape[0]
        return self.features

# # class Fib Time Series: functions for manipulating 
class FibTimeSeries:
    def __init__(self, cumlow, cummax, fib_levels, indices_extended = None):
        """basically a numpy array of fib retracement/extensions, plus some functions for extracting the information
        argments: cumlow/cummax are cumulative lows and highs
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1, 1.618, 2.618, 4.236, 6.854, 11.09]
        """
        if isinstance(fib_levels, list):
            self.fib_levels = np.array(fib_levels)
        elif isinstance(fib_levels, np.ndarray):
            self.fib_levels = fib_levels
        assert 'fib_levels' in dir(self)
        # size of series, within span
        self.n_inspan = cumlow.shape[0]
        self.n_fib_levels = len(self.fib_levels)
        assert self.n_inspan == cummax.shape[0]
        
        # time-indices
        self.index = cummax.index
        # ... time at beginning
        self.start_loc = self.index[0]
        self.end_loc = self.index[-1]
        
        # time series of retracements
        self.fib_series = self.calc_fibs(cumlow, cummax)
        self.shape = self.fib_series.shape
        
        # indices used for extended the series into the future
        self.indices_extended = indices_extended
    
    def calc_fib(self, lo, hi):
        """given a high, and a low, get the fibinocci extensions and retracements. """
        return (hi-lo)*self.fib_levels + lo
    
    def calc_fibs(self, cummax, cumlow):
        return np.array([self.calc_fib(hi,lo) for hi,lo in zip(cummax, cumlow)]).T
    
    def numpy(self):
        """default to returning the full series """
        return self.fib_series
    
    def __getitem__(self, item):
        """default getitem is to just return the numpy array __getitem__ """
        return self.fib_series[item]
    
    def get(self, start=None, end=None, return_indices = False):
        """elaborate getitem but allows time-index"""
        if start is None:
            if end is None:
                # case: if no indices specified, just return the numpy
                if return_indices:
                    return self.numpy(), self.index
                return self.numpy()
            start = self.start_loc
        if end is None:
            end = self.end_loc
        # how to do this?
        if (start <= self.end_loc):# and (end <= self.end_loc):
            # (notice we catch the edge case where start_loc == end_loc
            # macro-case, if the starting index is within (self.start_loc, self.end_loc)
            #indx_to_return_fibs = self.index[(self.index>=start) & (self.index<=end)]
            indx_to_return_fibs = np.where((self.index>=start) & (self.index<=end))[0]
            fib_series_to_return = self.fib_series[:,indx_to_return_fibs]
            # declare that for the extended data, the 'start' will be self.end_loc
            extended_start_loc =self.end_loc
        else:
            # macro-case, if the starting index is outside self.start_loc, self.end_loc)
            fib_series_to_return = np.array([[]]*self.n_fib_levels) # dummy to concatenate
            extended_start_loc = start + pd.Timedelta(-1, unit="day")
        if end > self.end_loc:
            # case: if the specified 'end' is greater than the time-series end-index,
            # ... then we must repeat the final fib_levels (at self.end_loc) for
            # ... the length of the extra time-indices
            # ... the extra time indices are NOT contiguous, so we need the user to set
            # ... self.indices_extended to know which dates are valid
            if self.indices_extended is None:
                raise ValueError("need to set 'self.indices_extended' to know for which dates to extrapolate the fib-extensions")
            indices_to_extrapolate_fibs = self.indices_extended[(self.indices_extended>extended_start_loc) & (self.indices_extended<=end)]
            # how much to extend
            n_extrapolate = len(indices_to_extrapolate_fibs)
            # extended series
            fib_series_extrapolate = np.array([self.fib_series[:,-1]]*n_extrapolate).T
        else:
            # none to extrapolate
            fib_series_extrapolate = np.array([[]]*self.n_fib_levels) # dummy to concatenate
        
        # concatenate the in-span and extrapolated
        return np.concatenate((fib_series_to_return, fib_series_extrapolate), axis=1)
    
    def mask_out_predrawdown(self, indx_start_of_credible_fib):
        """truncates the dataset for the initial drawdown where one couldn't have know that one would eventuallly be in a drawdown"""
        self.fib_series = self.fib_series[:,indx_start_of_credible_fib:]
        # correct the n_inspan
        self.n_inspan = self.fib_series.shape[-1]
        # correct the time-indices
        self.index = self.index[indx_start_of_credible_fib:]
        self.start_loc = self.index[0]
        # correct shape
        self.shape = self.fib_series.shape
    
