import time
import math
import scipy
import scipy.stats as stats
from scipy.stats import norm
golden_ratio = ((1 + 5 ** 0.5) / 2)-1
import sys
from datetime import datetime


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import pandas as pd
#import pandas_ta as ta

import random
import os
import os.path
from decimal import *

#%matplotlib inline
"""
import torch
import torch.nn as nn
import torch.utils.data as data

from IPython.display import set_matplotlib_formats
from IPython.display import clear_output
from tqdm.notebook import tqdm  # Progress bar

import plotly.graph_objects as go
"""
from matplotlib.colors import to_rgba


def do_it():
    print("Mi a hézag?")


print("np.__version__: ",np.__version__)
print("pd.__version__: ",pd.__version__)

print("hello te lóvacskaA")
print(golden_ratio)
a1 = (1+golden_ratio) - (1/golden_ratio)
fnum = 7.154327

a1 = "{:.17f}".format(a1)

print("a1: ",a1)
do_it()

class Calculus:
    def __init__(self, 
                 df, 
                 filter_limit = (-2,2),
                 sub_range_limit = (-1,1)
                ):
        ####################################################################
        self.df = df
        self.filter_limit_min, self.filter_limit_max = filter_limit
        self.is_long =   self.df["delta_price"] > self.filter_limit_max
        self.is_short =  self.df["delta_price"] < self.filter_limit_min
        self.df_long = self.df[self.is_long]
        self.df_short = self.df[self.is_short]
        
        self.sub_range_min, self.sub_range_max = sub_range_limit
        self.is_sub_range =  self.df["delta_price"] > self.sub_range_min 
        self.is_sub_range =  self.df["delta_price"] < self.sub_range_max
        self.df_sub_range = self.df[self.is_sub_range]
        
        ####################################################################
     
        self.np_all_delta_high = self.df["delta_high"].to_numpy()
        self.np_all_delta_low = self.df["delta_low"].to_numpy()

        self.np_long_delta_high = self.df_long["delta_high"].to_numpy()
        self.np_long_delta_low = self.df_long["delta_low"].to_numpy()

        self.np_short_delta_high = self.df_short["delta_high"].to_numpy()
        self.np_short_delta_low = self.df_short["delta_low"].to_numpy()
     
        self.np_sub_range_delta_high = self.df_sub_range["delta_high"].to_numpy()
        self.np_sub_range_delta_low = self.df_sub_range["delta_low"].to_numpy()

        ####################################################################
    
        self.y = self.np_all_delta_high
        self.count_all_high_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        self.y = self.np_all_delta_low
        self.count_all_low_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        self.y = self.np_long_delta_high
        self.count_long_high_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        self.y = self.np_long_delta_low
        self.count_long_low_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        self.y = self.np_short_delta_high
        self.count_short_high_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        self.y = self.np_short_delta_low
        self.count_short_low_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        self.y = self.np_sub_range_delta_high
        self.count_sub_range_high_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        self.y = self.np_sub_range_delta_low
        self.count_sub_range_low_delta =  np.array([ [item, np.count_nonzero(self.y == item) ]  for item in set(self.y) ])    

        ####################################################################
    
    def show_histogram(self, arg):
        plt.scatter(self.count_long_high_delta[:,0],self.count_long_high_delta[:,1], color= "orange")
        plt.scatter(self.count_long_low_delta[:,0],self.count_long_low_delta[:,1], color= "firebrick")

