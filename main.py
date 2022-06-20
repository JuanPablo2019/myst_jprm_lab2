"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Calculation and Time Series visualization of Market Microstructure indicators.             -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: JuanPablo2019                                                                               -- #
# -- license: GNU General Public License v3.                                                             -- #
# -- repository:https://github.com/JuanPablo2019/myst_jprm_lab1.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import data as dt
import functions as fn
import numpy as np
import pandas as pd
import itertools
#%% calcular midprice

data_1 = fn.f_descriptive_ob(dt.ob_data)
mid = data_1['midprice']

# keys en timestamp
l_ts =[pd.to_datetime(i) for i in  list(dt.ob_data.keys())][9:]
#%% count ocurrencies for each scenario

# e1 = el midprice_t = midprice_t+1 
# e2 = midprice_t != midprice_t+1

# formula P_t = E{P_t+1}
total = len(mid)-1
e1 =[ mid[i] == mid[i+1] for i in range(len(mid)-1)]
e2 = len(mid)-1 - sum(e1)

#save the results, counts and  proportions in a dictionary

exp_1 = {'e1':{'cantidad':sum(e1), 'proporcion':np.round(sum(e1)/total,2)},
            'e2':{'cantidad':e2, 'proporcion': np.round(e2/total,2)},
            'total':len(mid)-1 }

# imprimir los resultados
exp_1['e1']['proporcion']
exp_1['e2']['proporcion']
#%% Repeat (experiments for each minute)

#   Experiments 00:006-00:06:59...00:05:00-00:05:59
# only for the first hour

minutes = list(np.arange(0,60)) #counter

# set property (take unique values)
list(np.arange(0,60)) == list(set([i_ts.minute for i_ts in l_ts]))
# returns true, meaning that at least there is one ob per minute

#search for each minutes the orderbooks and calculate mid

import itertools

#getting the keys 
ts = dict(itertools.islice(dt.ob_data.items(),2401-9))
ts = list(ts.keys())
# calculating the mid as before
#mid2 = [(dt.ob_data[i]['ask'][0]+dt.ob_data[i]['bid'][0])*0.5 for i in ts.keys()]
mid2 = [(dt.ob_data[ts[i]]['ask'][0]+dt.ob_data[ts[i]]['bid'][0])*0.5 for i in range(0,len(ts))]
#creating an empty dataframe to save mid and timestamp
df = pd.DataFrame()
df['mid']=mid2
df['time']=[pd.to_datetime(i) for i in ts]
df['minute']=[i.minute for i in df['time'].tolist()]


#grouping in a dictionary the mids that happen in each minute
d = dict((i,list(df[df['minute']==i]['mid'])) for i in minutes)


#%% quantities

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

#%% Repeat experiment for the weighted midprice

w_mid = data_1['weighted_midprice']

#count ocurrencies for each scenario

# e1 = el w_midprice_t = w_midprice_t+1 
# e2 = w_midprice_t != w_midprice_t+1

# formula P_t = E{P_t+1}
total = len(w_mid)-1
e1_w =[ w_mid[i] == w_mid[i+1] for i in range(len(w_mid)-1)]
e2_w = len(w_mid)-1 - sum(e1_w)

#save the results, counts and  proportions in a dictionary

w_exp_1 = {'e1':{'cantidad':sum(e1_w), 'proporcion':np.round(sum(e1_w)/total,2)},
            'e2':{'cantidad':e2_w, 'proporcion': np.round(e2_w/total,2)},
            'total':len(w_mid)-1 }

# imprimir los resultados
# w_exp_1['e1']['proportion']
# w_exp_1['e2']['proportion']
#%% Repeat (experiments for each minute) weighted midprice

#   Experiments 00:006-00:06:59...00:05:00-00:05:59
# only for the first hour

#getting the keys 
ts_w = dict(itertools.islice(dt.ob_data.items(),2401-9))

# calculating the mid as before
w_mid2 = [dt.ob_data[i]['bid_size'].sum()/(dt.ob_data[i]['bid_size'].sum()+dt.ob_data[i]['ask_size'].sum())*(dt.ob_data[i]['ask'][0]+dt.ob_data[i]['bid'][0])*0.5 
          for i in ts.keys()]

#creating an empty dataframe to save mid and timestamp
df2 = pd.DataFrame()
df2['weighted_mid']=w_mid2
df2['time']=[pd.to_datetime(i) for i in ts]
df2['minute']=[i.minute for i in df2['time'].tolist()]


#grouping in a dictionary the mids that happen in each minute
d2 = dict((i,list(df2[df2['minute']==i]['weighted_mid'])) for i in minutes)


#%% quantities

e1_m2 = []
e2_m2 = []
prop2_e1 = []
prop2_e2 = []

for i in range(len(d2)):
    e1_t = np.sum([d2[i][j] == d2[i][j+1] for j in range(len(d2[i])-1)])
    e1_m2.append(e1_t)
    e2_t=len(d2[i])-1 - np.sum([d2[i][j] == d2[i][j+1] for j in range(len(d2[i])-1)])
    e2_m2.append(e2_t)
    total2 = len(d[i])-1
    prop2_e1.append(np.round(e1_t/total2,2))
    prop2_e2.append(np.round(e2_t/total2,2))
    
w_exp_2 = pd.DataFrame()
w_exp_2['interval']=minutes
w_exp_2['total']=[i+j for i,j in zip(e1_m2,e2_m2)]
w_exp_2['e1']=e1_m2
w_exp_2['e2']=e2_m2
w_exp_2['proportion e1'] = prop2_e1
w_exp_2['proportion e2'] = prop2_e2

#%% Function backtest autocovariance


pt = data_1['midprice']
spread = data_1['spread']
d_pt = [pt[i-1]-pt[i] for i in range(1,len(pt))]
gamma_1=fn.auto_cov(d_pt)
c = np.sqrt(gamma_1)
spread_pred = 2*c

#%% test for alll the pt series
gamma1=[]
pt = data_1['midprice']
d_pt = [pt[i-1]-pt[i] for i in range(1,len(pt))]

for i in range(2,len(d_pt)):
    xt = d_pt[0:i]
    gamma1.append(fn.auto_cov(xt))
    
c = np.array([np.sqrt(np.abs(i)) for i in gamma1])
spread_pred = 2*c
    