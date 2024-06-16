import pandas as pd
import pickle

with open('czgz_train.pkl', 'rb') as f:
    price_array = pickle.load(f)
out = pd.DataFrame(price_array)
out.to_csv("data_train.csv", index=False)