
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Calculation and Time Series visualization of Market Microstructure indicators.             -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: JuanPablo2019                                                                               -- #
# -- license: GNU General Public License v3.                                                             -- #
# -- repository:https://github.com/JuanPablo2019/myst_jprm_lab1.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
# libraries

import numpy as np
import data as dt
import pandas as pd
from scipy.stats import skew,kurtosis
import itertools


# imported data
data_ob=dt.ob_data


    
#---------------------- Orderbook Metrics----------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
def f_descriptive_ob(data_ob:dict) -> dict:
    """
    Docstring
  

    Parameters
    ----------
    data_ob : dict
        
        Orderbook as the input data, a dictionary with the following structure:
        "timestamp": object timestamp recognize by machine, e.g. pd.to_datetime()
        'bid_size:'volume for bid levels
        'bid:'bid price
        'ask:'ask price
        'ask_size: volume of ask levels
            

    Returns
    -------
    r_data: dict
        Dictionary with the following metrics. 
        'median_ts_ob':list containing float
        'midprice':list containing float
        'spread':list containing float
        'No. of levels':list containing int
        'Bid Volume':list containing float
        'Ask Volume': list containing float
        'Total Volume':list containing float
        'Orderbook Imbalance':list containing float
        'Weighted midprice':list containing float
        'VWAP Volume Weighted Average Price':list containing float
        'OHLCV': DataFrame of OHLCV sample by hour, shape [2,5] 
                columns: opening, close, minimun and maximun (calculated from orderbook midprice),
                         volume (calculated as the total volume)
        'stats_ob_imbalance': Dataframe containing the following statistical moments 
                             of the orderbook imbalance: Median, Variance, Bias, Kurtosis.
        'bid': bid price.
        'ask': ask price.
             
        
    
    -------

    """
   
    
# Median time of orderbook update
    ob_ts = list(data_ob.keys())
    l_ts =[ pd.to_datetime(i_ts) for i_ts in ob_ts]
    ob_m1 = np.median([l_ts[n_ts+1] - l_ts[n_ts] for n_ts in range(0,len(l_ts)-1)]).total_seconds()*1000
    
    
    # spread 
    ob_m2 = [data_ob[ob_ts[i]]['ask'][0]-data_ob[ob_ts[i]]['bid'][0] for i in range(0,len(ob_ts))]
    
    # mid price
    ob_m3 = [(data_ob[ob_ts[i]]['ask'][0]+data_ob[ob_ts[i]]['bid'][0])*0.5 for i in range(0,len(ob_ts))]
    
    # No. price levels
    ob_m4 = [data_ob[i_ts].shape[0] for i_ts in ob_ts]
   
    # bid volume
    ob_m5= [np.round(data_ob[i_ts]['bid_size'].sum(),6) for i_ts in ob_ts]
    
    #ask volumne
    ob_m6= [np.round(data_ob[i_ts]['ask_size'].sum(),6) for i_ts in ob_ts]
    
    #total volumne
    ob_m7= [np.round(data_ob[i_ts]['bid_size'].sum()+data_ob[i_ts]['ask_size'].sum(),6) for i_ts in ob_ts]
    
    #order book imbalance
    ob_imb = [data_ob[i]['bid_size'].sum()/(data_ob[i]['bid_size'].sum()+data_ob[i]['ask_size'].sum()) for i in ob_ts]
    

    # weighted midprice
    ob_wm = [ob_imb[i]*ob_m3[i] for i in range(0,len(ob_ts))]
    
    # Weighted midprice (B) (TOB) for extrapoints
    # W-MidPrice-B = [ask_volume/(total_volume)] *bid_price+[bid_volume/(total_volume)]*ask_price
    # W-MidPrice-B = (v[1]/np.sum(v[0]+v[1]))*p[0]+(v[0]/np.sum(v[0]+v[1]))*p[1]
    #VWAP Volume Weighted Average Price
    
    vwap = [np.sum(data_ob[i]['bid']*data_ob[i]['bid_size'] + data_ob[i]['ask']*data_ob[i]['ask_size'])/np.sum(data_ob[i]['bid_size']+data_ob[i]['ask_size'])
        for i in ob_ts]
    
    #OHLCV con mid price open, high, low, close, volume (Quoted volume)
    ohlcv = pd.DataFrame()
    ohlcv['midprice']=ob_m3
    ohlcv['Volume']=ob_m7
    ohlcv['Timestamp']=pd.to_datetime(ob_ts)
    ohlcv.set_index(['Timestamp'],inplace=True)
    
    ohlcv_hr = pd.DataFrame()
    ohlcv_hr['opening'] = ohlcv['midprice'].resample('60T').first()
    ohlcv_hr['close'] = ohlcv['midprice'].resample('60T').last()
    ohlcv_hr['min'] = ohlcv['midprice'].resample('60T').min()
    ohlcv_hr['max'] = ohlcv['midprice'].resample('60T').max()
    ohlcv_hr['volume'] = ohlcv['Volume'].resample('60T').sum()
    
    #-- (13) stats: Mediana, Varianza, Sesgo, Kurtosis for the ob_imb
    stats = pd.DataFrame({'Median': np.median(ob_imb),'Variance':np.var(ob_imb),
             'Skew':skew(ob_imb),'Kurtosis':kurtosis(ob_imb)},index=[1])
    #bid price
    bid = np.array([data_ob[ob_ts[i]]['bid'][0] for i in range(0,len(ob_ts))])
    
    #ask price
    ask = np.array([data_ob[ob_ts[i]]['ask'][0] for i in range(0,len(ob_ts))])
    
    r_data = {'median_ts_ob':ob_m1, 'spread': ob_m2, 'midprice': ob_m3, 
              'No. of price levels':ob_m4,'Bid Volume':ob_m5,
              'Ask Volume':ob_m6,'Total Volume':ob_m7,
              'orderbook_imbalance': ob_imb, 
              'weighted_midprice':ob_wm,
              'Volume Weighted Average Price':vwap,'OHLCV':ohlcv_hr,
              'stats_ob_imbalance':stats,'bid':bid,'ask':ask}
    

   
    
    
    
    
    return r_data


def f_martingale(data_ob:dict,price:str='midprice',interval:str='None') -> dict:
    """
    

    Parameters
    ----------
    data_ob : dict
        
        Orderbook as the input data, a dictionary with the following structure:
        "timestamp": object timestamp recognize by machine, e.g. pd.to_datetime()
        'bid_size:'volume for bid levels
        'bid:'bid price
        'ask:'ask price
        'ask_size: volume of ask levels
    
    price : str
         Define the type of price you want to test for your analysis pourpuse.
         By default it is set to work with the midprice, but it can also work
         weighted_midprice. 
         
         For the weighted midprice: the argument is 'weighted_midprice'
        For the midprice is: 'midprice'
        If the argument is not specify it would be calculated with the 
        midprice by default.
         
    interval : str
          The martingale property can be test with all the prices in the 
          Order Book or it can be done by minute interval for the first hour.
          For the interval in minutes the argument should be 'minutes'.
          If the argument is not given by default it would be calculated 
          with all the prices of the order book.


    Returns
    -------
    dict
        When the calculation is done without the minute interval, the function
        returns a dictionary with the following keys:
        
            e1: a dictionary with the number of ocurrencies where the 
                first scenario (e1) happen, and the proportion respect 
                to the total experiemnt trials.
                e1 = el w_midprice_t = w_midprice_t+1 
                
             e2: a dictionary with the number of ocurrencies where the 
                first scenario (e2) happen, and the proportion respect 
                to the total experiemnt trials.
                e2 = w_midprice_t != w_midprice_t+1
                
    DataFrame
        When the calculation is done by minute interval, the function returns
        a pandas dataframe compound of the following columns:
            
            interval: The minute 
            
            total: The sum of the experiments for each minute.
            
            e1: Number of trials that fullfill the first scenario.
            
            e2: Number of trials that fullfill the second scenario.
            
            proportion e1: The proportion of the first scenario that fullfill
            the first scenario respect to the total trails for each minute.
            
            proportion e2: The proportion of the second scenario that fullfill
            the first scenario respect to the total trails for each minute.
            

    """

       
   
    if price == 'midprice' and interval=='minutes':
        # only for the first hour
        # keys en timestamp
        #l_ts =[pd.to_datetime(i) for i in  list(data_ob.keys())][9:] 
        minutes = list(np.arange(0,60)) #indexer
       
       #search for each minutes the orderbooks and calculate mid

        #getting the keys 
        ts = dict(itertools.islice(data_ob.items(),2401-9))
        ts = list(ts.keys())
        # calculating the mid as before
        #mid2 = [(dt.ob_data[i]['ask'][0]+dt.ob_data[i]['bid'][0])*0.5 for i in ts.keys()]
        mid2 = [(data_ob[ts[i]]['ask'][0]+data_ob[ts[i]]['bid'][0])*0.5
                for i in range(0,len(ts))]
        
        #creating an empty dataframe to save mid and timestamp
        df = pd.DataFrame()
        df['mid']=mid2
        df['time']=[pd.to_datetime(i) for i in ts]
        df['minute']=[i.minute for i in df['time'].tolist()]
        
        
        #grouping in a dictionary the mids that happen in each minute
        d = dict((i,list(df[df['minute']==i]['mid'])) for i in minutes)
        
        #saving the experiment's results  (proportions, counts) in a dataframe
        e1_m = []
        e2_m = []
        prop_e1 = []
        prop_e2 = []
        
        for i in range(len(d)):
            e1_t = np.sum([d[i][j] == d[i][j+1] for j in range(0,len(d[i])-1)])
            e1_m.append(e1_t)
            e2_t=len(d[i])-1 - np.sum([d[i][j] == d[i][j+1] for j in range(0,len(d[i])-1)])
            e2_m.append(e2_t)
            total2 = len(d[i])-1
            prop_e1.append(np.round(e1_t/total2,2))
            prop_e2.append(np.round(e2_t/total2,2))
            
        exp_2 = pd.DataFrame()
        exp_2['interval']=minutes
        exp_2['total']=[i+j for i,j in zip(e1_m,e2_m)]
        exp_2['e1']=e1_m
        exp_2['e2']=e2_m
        exp_2['proportion e1'] = prop_e1
        exp_2['proportion e2'] = prop_e2
        
        return exp_2
    
    if price=='weighted_midprice' and interval=='minutes':
        
        minutes = list(np.arange(0,60)) #indexer
        ts = dict(itertools.islice(dt.ob_data.items(),2401-9))

        # calculating the mid as before
        w_mid = [data_ob[i]['bid_size'].sum()/(data_ob[i]['bid_size'].sum()+data_ob[i]['ask_size'].sum())*(data_ob[i]['ask'][0]+data_ob[i]['bid'][0])*0.5 
                  for i in ts.keys()]
        
        #creating an empty dataframe to save mid and timestamp
        df = pd.DataFrame()
        df['weighted_mid']=w_mid
        df['time']=[pd.to_datetime(i) for i in ts]
        df['minute']=[i.minute for i in df['time'].tolist()]


        #grouping in a dictionary the mids that happen in each minute
        d = dict((i,list(df[df['minute']==i]['weighted_mid'])) for i in minutes)
        
        #saving the experiment's results  (proportions, counts) in a dataframe

        e1 = []
        e2 = []
        prop_e1 = []
        prop_e2 = []
        
        for i in range(len(d)):
            e1_t = np.sum([d[i][j] == d[i][j+1] for j in range(len(d[i])-1)])
            e1.append(e1_t)
            e2_t=len(d[i])-1 - np.sum([d[i][j] == d[i][j+1] for j in range(len(d[i])-1)])
            e2.append(e2_t)
            total = len(d[i])-1
            prop_e1.append(np.round(e1_t/total,2))
            prop_e2.append(np.round(e2_t/total,2))
            
        exp_2 = pd.DataFrame()
        exp_2['interval']=minutes
        exp_2['total']=[i+j for i,j in zip(e1,e2)]
        exp_2['e1']=e1
        exp_2['e2']=e2
        exp_2['proportion e1'] = prop_e1
        exp_2['proportion e2'] = prop_e2
        
        return exp_2
    
    if price=='weighted_midprice' and interval=='None':
        
        ts = dict(itertools.islice(data_ob.items(),2401-9))

# calculating the mid as before
        w_mid = [data_ob[i]['bid_size'].sum()/(data_ob[i]['bid_size'].sum()+data_ob[i]['ask_size'].sum())*(data_ob[i]['ask'][0]+data_ob[i]['bid'][0])*0.5 
          for i in ts.keys()]
        
        
        #count ocurrencies for each scenario
        
        # e1 = el w_midprice_t = w_midprice_t+1 
        # e2 = w_midprice_t != w_midprice_t+1
        
        # formula P_t = E{P_t+1}
        total = len(w_mid)-1
        e1_w =[ w_mid[i] == w_mid[i+1] for i in range(len(w_mid)-1)]
        e2_w = len(w_mid)-1 - sum(e1_w)
        
        #save the results, counts and  proportions in a dictionary
        
        exp_1 = {'e1':{'cantidad':sum(e1_w), 'proporcion':np.round(sum(e1_w)/total,2)},
                    'e2':{'cantidad':e2_w, 'proporcion': np.round(e2_w/total,2)},
                    'total':len(w_mid)-1 }
        
        return exp_1
        
    
    else:
       ob_ts = list(data_ob.keys())
       #mid caluclation
       mid =  [(data_ob[ob_ts[i]]['ask'][0]+data_ob[ob_ts[i]]['bid'][0])*0.5 
               for i in range(0,len(ob_ts))]
       
       #martingale test
       total = len(mid)-1
       e1 =[ mid[i] == mid[i+1] for i in range(len(mid)-1)]
       e2 = len(mid)-1 - sum(e1)
       
       exp_1 = {'e1':{'cantidad':sum(e1), 'proporcion':np.round(sum(e1)/total,2)},
            'e2':{'cantidad':e2, 'proporcion': np.round(e2/total,2)},
            'total':len(mid)-1 }
       
       return exp_1
       
       
def auto_cov_delta(ts:list,lag:int=1) -> float:
    """
    The porpouse of this function is to calculate the autocovariance of
    a time series with n lags.

    Parameters
    ----------
    ts : list of fata (only the list without the timestamp)
        DESCRIPTION.
    lag : int, optional
        DESCRIPTION. The default is 1.
        

    Returns
    -------
    the autocovariance of the series with its k-lags. (float value)
     
    """
    delta = [ts[i-1]-ts[i] for i in range(1,len(ts))]
    mu = np.mean(delta)
    n = len(delta)
    r =[]
    for i in range(1,n):
        r.append((delta[i]-mu)*(delta[i-lag]-mu))


    return [sum(r)*(1/(n-lag)),delta]
           
         
def auto_corr_delta(ts:list,lag:int=1) -> float:
    """
    The porpouse of this function is to calculate the autocorrelation of
    a time series with n lags.

    Parameters
    ----------
    ts : list of fata (only the list without the timestamp)
        DESCRIPTION.
    lag : int, optional
        DESCRIPTION. The default is 1.
        

    Returns
    -------
    the autocovariance of the series with its k-lags. (float value)
     
    """
    delta = [ts[i-1]-ts[i] for i in range(1,len(ts))]
    mu = np.mean(delta)
    n = len(delta)
    r1 =[]
    r2=[]
    for i in range(1,n):
        r1.append((delta[i]-mu)*(delta[i-lag]-mu))
        r2.append((delta[i]-mu)**2)


    return   sum(r1)/sum(r2)

def roll_model(gamma1:float) -> float:
    """
    
    The objective of this function is to calculate the spread given by 
    the roll model.
    
    Parameters
    ----------
    gamma1 : float
             covariance of the changes of the variable of interest.
        

    Returns
    -------
    float
        The spread priced by the role model.

    """
    c = np.sqrt(np.abs(gamma1))

    model_spread = 2*c
    
    return model_spread

def roll_model_ts(ts: list)-> list:
    """
    The objective of this function is to calculate the spread priced by
    the role model for each point in time, in order to compare it with 
    another time series e.g. the real spread.

    Parameters
    ----------
    ts : list
         time series of the variable of interest
    
    

    Returns
    -------
    list
        the spread for each observation
        the length of the output would be of size ts-2
        because the autocorrelation function needs enough data to perform
        the necessary calculations.

    """
    delta = [ts[i-1]-ts[i] for i in range(1,len(ts))]
    gamma1_l=[]
    
    for i in range(2,len(delta)):
        xt = delta[0:i]
        mu = np.mean(xt)
        n = len(xt)
        r =[]
        for i in range(1,n):
                  r.append((xt[i]-mu)*(xt[i-1]-mu))
        gamma1_l.append(sum(r)*(1/(n-1)))
        
    c = np.array([np.sqrt(np.abs(i)) for i in gamma1_l])
    spread_pred = 2*c
    
    return spread_pred


    