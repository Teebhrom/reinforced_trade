from __future__ import annotations

from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import pickle

def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    if_trade=False,
    **kwargs,
):
    # import data processor
    from finrl.meta.data_processor import DataProcessor
    data_path = "pickle_collections/"
    
    #check if we have env_instance.pkl, if so, we can skip the data processing
    if if_trade:
        try:
            with open(data_path+"data_trade.pkl", "rb") as f:
                data = pickle.load(f)
            with open(data_path+"price_array_trade.pkl", "rb") as f:
                price_array = pickle.load(f)
            with open(data_path+"tech_array_trade.pkl", "rb") as f:
                tech_array = pickle.load(f)
            with open(data_path+"turbulence_array_trade.pkl", "rb") as f:
                turbulence_array = pickle.load(f)
            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_train": True,
            }
            env_instance = env(config=env_config)
            
        except:

            # fetch data
            dp = DataProcessor(data_source, **kwargs)
            data = dp.download_data(ticker_list, start_date, end_date, time_interval)
            data = dp.clean_data(data)
            data = dp.add_technical_indicator(data, technical_indicator_list)

            if if_vix:
                data = dp.add_vix(data)
            price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_train": False,
            }
            env_instance = env(config=env_config)
            with open("env_instance_test.pkl", "wb") as f:
                pickle.dump(env_instance, f)
            with open("data_test.pkl", "wb") as f:
                pickle.dump(data, f)
            with open("price_array_test.pkl", "wb") as f:
                pickle.dump(price_array, f)
            with open("tech_array_test.pkl", "wb") as f:
                pickle.dump(tech_array, f)
            with open("turbulence_array_test.pkl", "wb") as f:
                pickle.dump(turbulence_array, f)
    else:
        try:
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
        except:

            # fetch data
            dp = DataProcessor(data_source, **kwargs)
            data = dp.download_data(ticker_list, start_date, end_date, time_interval)
            data = dp.clean_data(data)
            data = dp.add_technical_indicator(data, technical_indicator_list)

            if if_vix:
                data = dp.add_vix(data)
            price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_train": False,
            }
            env_instance = env(config=env_config)
            with open("env_instance_test.pkl", "wb") as f:
                pickle.dump(env_instance, f)
            with open("data_test.pkl", "wb") as f:
                pickle.dump(data, f)
            with open("price_array_test.pkl", "wb") as f:
                pickle.dump(price_array, f)
            with open("tech_array_test.pkl", "wb") as f:
                pickle.dump(tech_array, f)
            with open("turbulence_array_test.pkl", "wb") as f:
                pickle.dump(turbulence_array, f)
    

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))

    #print("price_array: ", len(price_array))
    if drl_lib == "stable_baselines3":
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=cwd
        )
        
        # episode_data = DRLAgent_sb3.get_trajectory_load_from_file(
        #     model_name=model_name, environment=env_instance, cwd=cwd
        # )
        # with open("env_instance_test.pkl", "wb") as f:
        #     pickle.dump(env_instance, f)
        return episode_data
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty

    account_value_erl = test(
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="elegantrl",
        env=env,
        model_name="ppo",
        cwd="./test_ppo",
        net_dimension=512,
        kwargs=kwargs,
    )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # account_value_rllib = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo/checkpoint_000030/checkpoint-30",
    #     rllib_params=RLlib_PARAMS,
    # )
    #
    # # demo for stable baselines3
    # account_value_sb3 = test(
    #     start_date=TEST_START_DATE,
    #     end_date=TEST_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="sac",
    #     cwd="./test_sac.zip",
    # )
