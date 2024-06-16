from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
import pickle
import pandas as pd

kwargs = (
        {}
    ) 

data_source="yahoofinance"
ticker_list = DOW_30_TICKER


TRAIN_START_DATE = "2020-01-01" 
TRAIN_END_DATE = "2023-01-31"

TEST_START_DATE = "2023-08-28"
TEST_END_DATE = "2023-12-29"

TRADE_START_DATE = "2024-01-01"
TRADE_END_DATE = "2024-05-31"

if_vix = True

#for training data

start_date = TRAIN_START_DATE
end_date = TRAIN_END_DATE
time_interval = "1D"
technical_indicator_list = INDICATORS


dp = DataProcessor(data_source, **kwargs)
data = dp.download_data(ticker_list, start_date,  end_date, time_interval)
data = dp.clean_data(data)
data = dp.add_technical_indicator(data, technical_indicator_list)

if if_vix :
    data = dp.add_vix(data)
price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

with open("data_train.pkl", "wb") as f:
    pickle.dump(data, f)
with open("price_array_train.pkl", "wb") as f:
    pickle.dump(price_array, f)
with open("tech_array_train.pkl", "wb") as f:
    pickle.dump(tech_array, f)
with open("turbulence_array_train.pkl", "wb") as f:
    pickle.dump(turbulence_array, f)

#for test data

start_date = TEST_START_DATE
end_date = TEST_END_DATE
time_interval = "1D"
technical_indicator_list = INDICATORS

dp = DataProcessor(data_source, **kwargs)
data = dp.download_data(ticker_list, start_date,  end_date, time_interval)
data = dp.clean_data(data)
data = dp.add_technical_indicator(data, technical_indicator_list)

if if_vix :
    data = dp.add_vix(data)
price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

with open("data_test.pkl", "wb") as f:
    pickle.dump(data, f)
with open("price_array_test.pkl", "wb") as f:
    pickle.dump(price_array, f)
with open("tech_array_test.pkl", "wb") as f:
    pickle.dump(tech_array, f)
with open("turbulence_array_test.pkl", "wb") as f:
    pickle.dump(turbulence_array, f)

# #for trade data

start_date = TRADE_START_DATE
end_date = TRADE_END_DATE
time_interval = "1D"
technical_indicator_list = INDICATORS

dp = DataProcessor(data_source, **kwargs)
data = dp.download_data(ticker_list, start_date,  end_date, time_interval)
data = dp.clean_data(data)
data = dp.add_technical_indicator(data, technical_indicator_list)

if if_vix :
    data = dp.add_vix(data)
price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

with open("data_trade.pkl", "wb") as f:
    pickle.dump(data, f)
with open("price_array_trade.pkl", "wb") as f:
    pickle.dump(price_array, f)
with open("tech_array_trade.pkl", "wb") as f:
    pickle.dump(tech_array, f)
with open("turbulence_array_trade.pkl", "wb") as f:
    pickle.dump(turbulence_array, f)



