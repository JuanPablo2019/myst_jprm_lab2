
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
    book: dictionary of an orderbook compound of the following variables:
          bid_size, bid, ask_size, ask.
    Returns
    -------
    A plotly graph with the orderbooks levels (bid/ask) X axis. 
    Y axis represent the volume.
    
  
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
    Univariate Boxplot function with plotly.

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
    

def boxplot_multi(df:dict, variables:list, x_ax:str, orient:str, title:str, 
                  h:int, xaxes:str, yaxes:str, newnames:dict):
    """
    

    Returns
    -------
    df: dict
        dataframe (two dimensional dictionary) with the variables of interest.
        
    variables: list
               variables or variable you want to plot a list of strings.
    x_ax: str
          the variable that will represent the x axi in the plot.
          
    oreint: str
            chart orientation v for vertical, h for horizontal.
            
    title: str
           The title of the chart.
           
    h:    int
          height of the chart.
          
    xaxes: str
           the name of the x axis.
           
    yaxes: str
           the name of the y axis.
    
    newnames: dict
              names for the variables that are going to be visible.
    """

    fig = px.bar(df, y=variables, x=x_ax ,orientation=orient,
             title=title,height=h,width=1000)
    fig.update_layout(barmode='relative', bargap=0.50,bargroupgap=0.0)
    
    
    fig.for_each_trace(lambda t: t.update(name = newnames[t.name]))
    
    fig.update_xaxes(title=xaxes)
    fig.update_yaxes(title=yaxes)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 0,
            dtick = 1
        )
    )
    return fig.show()