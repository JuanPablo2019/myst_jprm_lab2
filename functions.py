
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
#from re import L
import numpy as np
import data as dt
import pandas as pd
from scipy.stats import skew,kurtosis


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
    
    r_data = {'median_ts_ob':ob_m1, 'spread': ob_m2, 'midprice': ob_m3, 
              'No. of price levels':ob_m4,'Bid Volume':ob_m5,
              'Ask Volume':ob_m6,'Total Volume':ob_m7,
              'orderbook_imbalance': ob_imb, 
              'weighted_midprice':ob_wm,
              'Volume Weighted Average Price':vwap,'OHLCV':ohlcv_hr,
              'stats_ob_imbalance':stats}
    

   
    
    
    
    
    return r_data