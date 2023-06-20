import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from ta.trend import MACD
from ta.momentum import StochasticOscillator

class Market_Data():

    def __init__(self, ticker, period, interval):
        
        self.ticker = ticker
        self.period = period
        self.interval = interval

        # Get the data set
        self.data = yf.download(tickers=ticker, period=period, interval=interval)

        # Create figure
        # add subplot properties when initializing fig variable
        self.other_figs = make_subplots(rows=4, cols=1,
                                         shared_xaxes=True,
                                         vertical_spacing=.02,
                                         row_heights=[0.6,0.3,0.2,0.2])


    # Print the table
    def print_data(self):
        print(self.data)

    # This will make it eaiser than reading the csv you will have direct access to the parts like what I have highlighted
    # I forgot to add this method
    def get_data(self):
        return self.data


    def bollinger_bands(self):
        # 20 day sma
        self.data['20SMA'] = self.data['Close'].rolling(20).mean()
        std = self.data['20SMA'].rolling(10).std()

        sma = self.data['20SMA']

        # Calculate Bands
        bollinger_uppper = sma + std * 2
        bollinger_lower = sma - std * 2 

        # Set the bands in the table
        self.data['BollingerU'] = bollinger_uppper
        self.data['BollingerL'] = bollinger_lower

        #Add Bollinger on the graph
        self.other_figs.add_trace(go.Scatter(x=self.data.index, 
                                y= self.data['BollingerU'],
                                line=dict(color='blue', width=1.5),
                                 name = 'Upper Bollinger'))

        self.other_figs.add_trace(go.Scatter(x=self.data.index,
                                 y= self.data['BollingerL'],
                                 line=dict(color='orange', width=1.5),
                                 name = 'Lower Bollinger'))


    def moving_averages(self):
        self.data['MA20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA5'] = self.data['Close'].rolling(window=5).mean()

        # Add the moving averages
        self.other_figs.add_trace(go.Scatter(x=self.data.index, 
                         y=self.data['MA5'], 
                         opacity=0.7, 
                         line=dict(color='blue', width=2), 
                         name='MA 5'), row=2, col=1)
        self.other_figs.add_trace(go.Scatter(x=self.data.index, 
                         y=self.data['MA20'], 
                         opacity=0.7, 
                         line=dict(color='orange', width=2), 
                         name='MA 20'), row=2, col=1)

    # MACD
    def macd(self):

        # MACD
        macd = MACD(close=self.data['Close'], 
                    window_slow=26,
                    window_fast=12, 
                    window_sign=9)

        # Plot MACD trace on 3rd row
        self.other_figs.add_trace(go.Bar(x=self.data.index, 
                            y=macd.macd_diff()
                            ), row=3, col=1)
        self.other_figs.add_trace(go.Scatter(x=self.data.index,
                                y=macd.macd(),
                                line=dict(color='black', width=2)
                                ), row=3, col=1)
        self.other_figs.add_trace(go.Scatter(x=self.data.index,
                                y=macd.macd_signal(),
                                line=dict(color='blue', width=1)
                                ), row=3, col=1)
        

    # RSI can be set to simple or expontinal rsi
    def rsi(self, periods=14, simple=True):
        
        closing_delta = self.data['Close'].diff()

        # Get upper and lower bands
        up = closing_delta.clip(lower=0)
        down = -1 * closing_delta.clip(upper=0)

        if simple == False:
            # Exponential moving average
            moving_average_up = up.ewm(com = periods - 1, min_periods = periods).mean()
            moving_average_down = down.ewm(com = periods - 1, min_periods = periods).mean()
        else:
            # Simple moving average
            moving_average_up = up.rolling(window = periods).mean()
            moving_average_down = down.rolling(window = periods).mean()
        

        rsi = moving_average_up / moving_average_down
        rsi = 100 - (100/(1 + rsi))

        self.data['RSI'] = rsi

        # Add to the graph
        self.other_figs.add_trace(go.Scatter(x=self.data.index, y= self.data['RSI'], line=dict(color='grey', width=1.5), name = 'RSI'), row=4, col=1)

    
    def create_figure(self):

        # Create the candle sticks
        self.other_figs.add_trace(go.Candlestick(x=self.data.index,
                        open=self.data['Open'],
                        high=self.data['High'],
                        low=self.data['Low'],
                        close=self.data['Close'], name = 'market data'))

        # remove rangeslider
        self.other_figs.update_layout(xaxis_rangeslider_visible=False)

        # change labels
        self.other_figs.update_yaxes(title_text="Price", row=1, col=1)
        self.other_figs.update_yaxes(title_text="Moving Average", row=2, col=1)
        self.other_figs.update_yaxes(title_text="MACD", showgrid=False, row=3, col=1)
        self.other_figs.update_yaxes(title_text="RSI", row=4, col=1)


    def show_figure(self):
        # Present figure
        self.other_figs.show()

    
    def get_table_values_csv(self):

        self.data.to_csv('data.csv')


if __name__ == '__main__':
    data = Market_Data('BTC-USD', '5d', '5m')

    # Adds the market indicatiors
    data.bollinger_bands()
    # Not sure what the difference between simple or exp
    data.rsi(simple=False)
    data.macd()
    data.moving_averages()

    # Creates and shows the figure
    data.create_figure()
    data.show_figure()

    # Prints out the data table
    data.print_data()

    data.get_table_values_csv()
