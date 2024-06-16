import pandas as pd
import pickle
from finrl.

with open('data_trade.pkl', 'rb') as f:
    data = pickle.load(f)
with open('price_array_trade.pkl', 'rb') as f:
    price_array = pickle.load(f)
with open('tech_array_trade.pkl', 'rb') as f:
    tech_array = pickle.load(f)
with open('turbulence_array_trade.pkl', 'rb') as f:
    turbulence_array = pickle.load(f)
 