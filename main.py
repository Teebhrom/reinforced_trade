from __future__ import annotations

import os
from argparse import ArgumentParser
from typing import List

from finrl.config import ALPACA_API_BASE_URL
from finrl.config import DATA_SAVE_DIR
from finrl.config import ERL_PARAMS
from finrl.config import SAC_PARAMS
from finrl.config import PPO_PARAMS
from finrl.config import TD3_PARAMS
from finrl.config import DDPG_PARAMS
from finrl.config import A2C_PARAMS
from finrl.config import INDICATORS
from finrl.config import RESULTS_DIR
from finrl.config import TENSORBOARD_LOG_DIR
from finrl.config import TEST_END_DATE
from finrl.config import TEST_START_DATE
from finrl.config import TRADE_END_DATE
from finrl.config import TRADE_START_DATE
from finrl.config import TRAIN_END_DATE
from finrl.config import TRAIN_START_DATE
from finrl.config import TRAINED_MODEL_DIR
from finrl.config_tickers import DOW_30_TICKER
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
import pickle

# construct environment

# try:
#     from finrl.config_private import ALPACA_API_KEY, ALPACA_API_SECRET
# except ImportError:
#     raise FileNotFoundError(
#         "Please set your own ALPACA_API_KEY and ALPACA_API_SECRET in config_private.py"
#     )


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


# "./" will be added in front of each directory
def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)


def main() -> int:
    parser = build_parser()
    options = parser.parse_args()
    check_and_make_directories(
        [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
    )

    if options.mode == "train":
        from finrl import train

        env = StockTradingEnv

        # demo for elegantrl
        kwargs = (
            {}
        )  # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        try:
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
                    cwd="./trained_models/test_sac",
                    agent_params=SAC_PARAMS,
                    break_step=1e4,
                    kwargs=kwargs,
                )
        except:
            print("error in training sac")
        try:
            train(
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                ticker_list=DOW_30_TICKER,
                data_source="yahoofinance",
                time_interval="1D",
                technical_indicator_list=INDICATORS,
                drl_lib="stable_baselines3",
                env=env,
                model_name="ppo",
                cwd="./trained_models/test_ppo",
                agent_params=PPO_PARAMS,
                break_step=1e4,
                kwargs=kwargs,
            )
            
        except:
            print("error in training ppo")
        try:
            train(
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                ticker_list=DOW_30_TICKER,
                data_source="yahoofinance",
                time_interval="1D",
                technical_indicator_list=INDICATORS,
                drl_lib="stable_baselines3",
                env=env,
                model_name="ddpg",
                cwd="./trained_models/test_ddpg",
                agent_params=DDPG_PARAMS,
                break_step=1e4,
                kwargs=kwargs,
            )
        except:
            print("error in training ddpg")
        try:
            train(
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                ticker_list=DOW_30_TICKER,
                data_source="yahoofinance",
                time_interval="1D",
                technical_indicator_list=INDICATORS,
                drl_lib="stable_baselines3",
                env=env,
                model_name="td3",
                cwd="./trained_models/test_td3",
                agent_params=TD3_PARAMS,
                break_step=1e4,
                kwargs=kwargs,
            )
        except:
            print("error in training td3")
        try:
            train(
                start_date=TRAIN_START_DATE,
                end_date=TRAIN_END_DATE,
                ticker_list=DOW_30_TICKER,
                data_source="yahoofinance",
                time_interval="1D",
                technical_indicator_list=INDICATORS,
                drl_lib="stable_baselines3",
                env=env,
                model_name="a2c",
                cwd="./trained_models/test_a2c",
                agent_params=A2C_PARAMS,
                break_step=1e4,
                kwargs=kwargs,
            )
        except:
            print("error in training a2c")
        
    elif options.mode == "test":
        from finrl import test

        env = StockTradingEnv

        # demo for elegantrl
        # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        kwargs = {}

        account_value_erl = test(  # noqa
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ddpg",
            cwd="./trained_models/test_ddpg",
            net_dimension=512,
            kwargs=kwargs,
        )
    elif options.mode == "trade":
        from finrl import test

        env = StockTradingEnv

        # demo for elegantrl
        # in current meta, with respect yahoofinance, kwargs is {}. For other data sources, such as joinquant, kwargs is not empty
        kwargs = {}

        account_value_erl = test(  # noqa
            start_date=TRADE_START_DATE,
            end_date=TRADE_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ddpg",
            cwd="./trained_models/test_ddpg",
            net_dimension=512,
            if_trade=True,
            kwargs=kwargs,
        )
    else:
        raise ValueError("Wrong mode.")
    return 0


# Users can input the following command in terminal
# python main.py --mode=train
# python main.py --mode=test
# python main.py --mode=trade
if __name__ == "__main__":
    raise SystemExit(main())
