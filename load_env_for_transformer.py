import numpy as np

import collections
import pickle

from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv


datasets = []

env = StockTradingEnv
data_path = "pickle_collections/"
with open(data_path+"data_test.pkl", "rb") as f:
    data = pickle.load(f)
with open(data_path+"price_array_test.pkl", "rb") as f:
    price_array = pickle.load(f)
with open(data_path+"tech_array_test.pkl", "rb") as f:
    tech_array = pickle.load(f)
with open(data_path+"turbulence_array_test.pkl", "rb") as f:
    turbulence_array = pickle.load(f)
env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": True,
}
env_instance = env(config=env_config)

with open("env_instance_test.pkl", "wb") as f:
    pickle.dump(env_instance, f)
    
    
    
with open(data_path+"data_train.pkl", "rb") as f:
    data = pickle.load(f)
with open(data_path+"price_array_train.pkl", "rb") as f:
    price_array = pickle.load(f)
with open(data_path+"tech_array_train.pkl", "rb") as f:
    tech_array = pickle.load(f)
with open(data_path+"turbulence_array_train.pkl", "rb") as f:
    turbulence_array = pickle.load(f)
env_config = {
    "price_array": price_array,
    "tech_array": tech_array,
    "turbulence_array": turbulence_array,
    "if_train": True,
}
env_instance = env(config=env_config)

with open(data_path+"env_instance_train.pkl", "wb") as f:
    pickle.dump(env_instance, f)