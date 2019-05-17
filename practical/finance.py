import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as web

info = web.DataReader('F', data_source='iex', start=    '1/2/2016', end='31/12/2017')