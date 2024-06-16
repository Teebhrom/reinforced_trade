import matplotlib.pyplot as plt
import pandas as pd
import pickle
from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from finrl.meta.data_processor import DataProcessor
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from min_decision_transformer.decision_transformer.model import DecisionTransformer

from min_decision_transformer.dataset.D4RLTrajectoryDataset import D4RLTrajectoryDataset, evaluate_on_env, get_total_asset_list


rtg_scale = 1              # scale to normalize returns to go
rtg_target = 1000
# use v3 env for evaluation because
# DT paper evaluates results on v3 envs

env_name = 'stocktrading'

max_eval_ep_len = 1000      # max len of one evaluation episode
num_eval_ep = 5           # num of evaluation episodes per iteration

batch_size = 20            # training batch size
lr = 1e-5                  # learning rate
wt_decay = 1e-4             # weight decay
warmup_steps = 10000        # warmup steps for lr scheduler

# total updates = max_train_iters x num_updates_per_iter
max_train_iters = 500 #200
num_updates_per_iter = 100

context_len = 20        # K in decision transformer
n_blocks = 3            # num of transformer blocks
embed_dim = 32         # embedding (hidden) dim of transformer
n_heads =  2          # num of transformer heads
dropout_p = 0.1         # dropout probability


rtg_scale = 1              # scale to normalize returns to go
rtg_target = 1000
# use v3 env for evaluation because
# DT paper evaluates results on v3 envs

env_name = 'stocktrading'

max_eval_ep_len = 1000      # max len of one evaluation episode
num_eval_ep = 5           # num of evaluation episodes per iteration

batch_size = 10            # training batch size
lr = 1e-4                  # learning rate
wt_decay = 1e-4             # weight decay
warmup_steps = 10000        # warmup steps for lr scheduler

# total updates = max_train_iters x num_updates_per_iter
max_train_iters = 500 #200
num_updates_per_iter = 100

context_len = 10        # K in decision transformer
n_blocks = 3            # num of transformer blocks
embed_dim = 100         # embedding (hidden) dim of transformer
n_heads =  1          # num of transformer heads
dropout_p = 0.1         # dropout probability

for mode in [ 'test', 'trade']:
    pickle_path = 'pickle_collections/'
    #pickle_path = ''

    with open(pickle_path+'data_'+mode+'.pkl', "rb") as f:
        data = pickle.load(f)
    with open(pickle_path+'price_array_'+mode+'.pkl', "rb") as f:
        price_array = pickle.load(f)
    with open(pickle_path+'tech_array_'+mode+'.pkl', "rb") as f:
        tech_array = pickle.load(f)
    with open(pickle_path+'turbulence_array_'+mode+'.pkl', "rb") as f:
        turbulence_array = pickle.load(f)
    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = StockTradingEnv(config=env_config)

    agent = DRLAgent_sb3(env = env_instance)

    model_name_list = ['a2c','ppo','td3','sac','ddpg']
    model_path_list = ['./test_a2c','./test_ppo','./test_td3','./test_sac','./test_ddpg']

    plot_path = 'plots/'

    plt.figure(figsize = (15, 7))
    plt.title('Total Asset Value')

    state_dim = env_instance.observation_space.shape[0]
    act_dim = env_instance.action_space.shape[0]

    model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                n_blocks=n_blocks,
                h_dim=embed_dim,
                context_len=context_len,
                n_heads=n_heads,
                drop_p=dropout_p,
            )




    model.load_state_dict(torch.load('log_dt/dt_stock_trading_model_24-06-09-21-35-04_best.pt'))


    total_asset_lists = get_total_asset_list(model, 'cpu', context_len, env_instance, rtg_target, rtg_scale,
                        max_test_ep_len=1000,
                        state_mean=None, state_std=None)

    # reward = evaluate_on_env(model, 'cpu', context_len, env_instance, rtg_target, rtg_scale,
    #                     max_test_ep_len=1000,
    #                     state_mean=None, state_std=None)
    # print(reward)


    date = data["timestamp"].unique()

    plt.plot(date, total_asset_lists)
    plt.xlabel('Date')

    dataframe = pd.DataFrame()
    dataframe['date'] = date
    dataframe['transformer'] = total_asset_lists

    for model_name, model_path in zip(model_name_list, model_path_list):
        env_instance = StockTradingEnv(config=env_config)
        total_asset_lists = agent.DRL_prediction_load_from_file(model_name=model_name, environment=env_instance, cwd="trained_models/"+model_path)
        dataframe[model_name] = total_asset_lists
        plt.plot(date, total_asset_lists)
        
        
    plt.legend(['transformer']+model_name_list)    

    plt.savefig(plot_path +'Total Asset Value ' + mode +'.png')

    dataframe.to_csv('Total Asset Value ' + mode +' .csv', index=False)