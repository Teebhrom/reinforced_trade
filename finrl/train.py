from __future__ import annotations

from finrl.config import ERL_PARAMS
from finrl.config import INDICATORS
from finrl.config import RLlib_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
import pickle

# construct environment


def train(
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
    **kwargs,
):
    
    data_path = "pickle_collections/"
    #check if we have env_instance.pkl, if so, we can skip the data processing
    try:
        with open(data_path+"data_train.pkl", "rb") as f:
            print("data_train found, skipping data processing")
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
        env_instance = env(config=env_config, turbulence_thresh = 800)
    except:
    # download data
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
            "if_train": True,
        }
        
        with open("data.pkl", "wb") as f:
            pickle.dump(data, f)
        with open("price_array.pkl", "wb") as f:
            pickle.dump(price_array, f)
        with open("tech_array.pkl", "wb") as f:
            pickle.dump(tech_array, f)
        with open("turbulence_array.pkl", "wb") as f:
            pickle.dump(turbulence_array, f)
        env_instance = env(config=env_config)
        with open("env_instance.pkl", "wb") as f:  
            pickle.dump(env_instance, f)
    

    # read parameters
    cwd = kwargs.get("cwd", "./trained_models/" + str(model_name))

    if drl_lib == "stable_baselines3":
        total_timesteps = kwargs.get("total_timesteps", 3e5)
        agent_params = kwargs.get("agent_params")
        from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

        agent = DRLAgent_sb3(env=env_instance)
        model = agent.get_model(model_name, model_kwargs=agent_params, tensorboard_log = TENSORBOARD_LOG_DIR)
        trained_model = agent.train_model(
            model=model, tb_log_name=model_name, total_timesteps=total_timesteps
        )
        print("Training is finished!")
        trained_model.save(cwd)
        print("Trained model is saved in " + str(cwd))
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")


if __name__ == "__main__":
    env = StockTradingEnv

    # demo for elegantrl
    kwargs = (
        {}
    )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="stable_baselines3",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     erl_params=ERL_PARAMS,
    #     break_step=1e5,
    #     kwargs=kwargs,
    # )

    ## if users want to use rllib, or stable-baselines3, users can remove the following comments

    # # demo for rllib
    # import ray
    # ray.shutdown()  # always shutdown previous session if any
    # train(
    #     start_date=TRAIN_START_DATE,
    #     end_date=TRAIN_END_DATE,
    #     ticker_list=DOW_30_TICKER,
    #     data_source="yahoofinance",
    #     time_interval="1D",
    #     technical_indicator_list=INDICATORS,
    #     drl_lib="rllib",
    #     env=env,
    #     model_name="ppo",
    #     cwd="./test_ppo",
    #     rllib_params=RLlib_PARAMS,
    #     total_episodes=30,
    # )
    #
    # # demo for stable-baselines3
    train(
        start_date=TRAIN_START_DATE,
        end_date=TRAIN_END_DATE,
        ticker_list=DOW_30_TICKER,
        data_source="yahoofinance",
        time_interval="1D",
        technical_indicator_list=INDICATORS,
        drl_lib="stable_baselines3",
        env=env,
        model_name="sac",
        cwd="./test_sac",
        agent_params=SAC_PARAMS,
        total_timesteps=1e4,
    )
