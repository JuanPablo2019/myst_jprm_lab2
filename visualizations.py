
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Calculation and Time Series visualization of Market Microstructure indicators.             -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: JuanPablo2019                                                                               -- #
# -- license: GNU General Public License v3.                                                             -- #
# -- repository:https://github.com/JuanPablo2019/myst_jprm_lab1.git                                      -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import plotly.offline as pyo
import pandas as pd
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
#---------------OrderBook Plot------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------

def plot_orderbook(book):
    """
    Limit OrderBook horizontal bars plot.
    
    Parameters
    ----------
    Returns
    -------
    
    References
    ----------
    """
    buy_side=book[['bid_size', 'bid']]
    buy_side=buy_side.groupby(['bid']).sum()
    buy_side['side']='buy'
    sell_side=book[['ask_size', 'ask']]
    sell_side=sell_side.groupby(['ask']).sum()
    sell_side['side']='sell'
    
    price_bid = list(buy_side.index)
    price_ask = list(sell_side.index)
    price_levels = price_bid + price_ask
    
    
    s_bid = list(buy_side['bid_size'])
    s_ask = list(sell_side['ask_size'])
    size_levels = s_bid + s_ask
    
    
    b_side = list(buy_side['side'])
    a_side = list(sell_side['side'])
    side_levels = b_side + a_side
    
    
    ob = pd.DataFrame()
    ob['price']=price_levels
    ob['size']=size_levels
    ob['side']=side_levels
    
    
    
  
    fig = px.bar(ob,x='price',y='size',color='side')
    return fig.show()


def plot_ts(y:list, x:list, variable_name:str):
    """
    Univariate time series plot function. 

    Parameters
    ----------
    y : list
        which contain the values of the variable of interest.
    x : list 
        which coitain the timestamp.
        
    variable_name : str
        The name you want to use for your variable of interest.
    title : str
        DESCRIPTION.

    Returns
    -------
    plotly plot.
  

    """
    
    
    
    df = pd.DataFrame()
    df[variable_name]=y
    df['time']=x
    
    fig = px.line(df, x='time', y=variable_name)
    
    return  fig.show()

def plot_boxplot(y:list, variable_name:str):
    """
    Boxplot function.

    Parameters
    ----------
    y : list
         which contain the values of the variable of interest.
    variable_name : str
         The name you want to use for your variable of interest.

    Returns
    -------
    plotly boxplot.

    """
    df = pd.DataFrame()
    df[variable_name]=y
    
    fig = px.box(df, y=variable_name, title=variable_name +'Boxplot')
    
    return fig.show()
    