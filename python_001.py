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
import torch
import torch.nn as nn
import torch.utils.data as data

import pandas as pd
import pandas_ta as ta

import random
import os
import os.path

%matplotlib inline
from IPython.display import set_matplotlib_formats
from IPython.display import clear_output
from matplotlib.colors import to_rgba
from tqdm.notebook import tqdm  # Progress bar

import plotly.graph_objects as go

sigmuid_torch = nn.Sigmoid()

set_matplotlib_formats("svg", "pdf")
print("Using torch", torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#device = "cpu"
print("torch.device: ", device)
