
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: APT and Roll Model apply to Market Microstructue.                                          -- #
# -- script: data.py : python script for data collection                                                 -- #
# -- author: JuanPablo2019                                                                               -- #
# -- license: GNU General Public License v3.                                                             -- #
# -- repository:https://github.com/JuanPablo2019/myst_jprm_lab2.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""


#%% libraries

import pandas as pd
import json



#%% Orderbook data

f = open('files/orderbooks_05jul21.json')
orderbooks_data = json.load(f)

#%% orderbook data wranging

ob_data = orderbooks_data['bitfinex']

ob_data = {i_key: i_value for i_key, i_value in ob_data.items() if i_value is not None}

ob_data = {i_ob: pd.DataFrame(ob_data[i_ob])[['bid_size','bid','ask','ask_size']]
                   if ob_data[i_ob] is not None else None for i_ob in list(ob_data.keys())}
