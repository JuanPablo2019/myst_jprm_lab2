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
#%% calcular midprice

data_1 = fn.f_descriptive_ob(dt.ob_data)
mid = data_1['midprice']

# keys en timestamp
l_ts =[pd.to_datetime(i) for i in  list(dt.ob_data.keys())][9:]
#%% Contabilizar ocurrencias de escenarios

# e1 = el midprice_t = midprice_t+1 
# e2 = midprice_t != midprice_t+1

# formula P_t = E{P_t+1}
total = len(mid)-1
e1 =[ mid[i] == mid[i+1] for i in range(len(mid)-1)]
e2 = len(mid)-1 - sum(e1)

#guardar resultados conteo y proporción en un diccionario

exp_1 = {'e1':{'cantidad':sum(e1), 'proporcion':np.round(sum(e1)/total,2)},
            'e2':{'cantidad':e2, 'proporcion': np.round(e2/total,2)},
            'total':len(mid)-1 }

# imprimir los resultados
exp_1['e1']['proporcion']
exp_1['e2']['proporcion']
#%% Repetir lo anterior para otros (expeerimentos de cada minuto)

#   Experimentos 00:006-00:06:59...00:05:00-00:05:59
# only for the first hour

minutes = list(np.arange(0,60)) #counter

# set property (take unique values)
list(np.arange(0,60)) == list(set([i_ts.minute for i_ts in l_ts]))
# returns true, meaning that at least there is one ob per minute

#search for each minutes the orderbooks and calculate mid

import itertools

#getting the keys 
ts = dict(itertools.islice(dt.ob_data.items(),2401-9))

# calculating the mid as before
mid2 = [(dt.ob_data[i]['ask'][0]+dt.ob_data[i]['bid'][0])*0.5 for i in ts.keys()]

#creating an empty dataframe to save mid and timestamp
df = pd.DataFrame()
df['mid']=mid2
df['time']=[pd.to_datetime(i) for i in ts]
df['minute']=[i.minute for i in df['time'].tolist()]


#grouping in a dictionary the mids that happen in each minute
d = dict((i,list(df[df['minute']==i]['mid'])) for i in minutes)


#%% cantidades

e1_m = []
e2_m = []
prop_e1 = []
prop_e2 = []

for i in range(len(d)):
    e1_t = np.sum([d[i][j] == d[i][j+1] for j in range(len(d[i])-1)])
    e1_m.append(e1_t)
    e2_t=len(d[i])-1 - np.sum([d[i][j] == d[i][j+1] for j in range(len(d[i])-1)])
    e2_m.append(e2_t)
    total2 = len(d[i])-1
    prop_e1.append(np.round(e1_t/total2,2))
    prop_e2.append(np.round(e2_t/total2,2))
    
exp_2 = pd.DataFrame()
exp_2['minute']=minutes
exp_2['e1']=e1_m
exp_2['e2']=e2_m
exp_2['proportion e1'] = prop_e1
exp_2['proportion e2'] = prop_e2
