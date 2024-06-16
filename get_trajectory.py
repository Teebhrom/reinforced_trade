from finrl.meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
import pickle
from finrl.meta.data_processor import DataProcessor
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3

models_name = ['a2c','ppo','td3','sac','ddpg']

models_path = ['./test_a2c','./test_ppo','./test_td3','./test_sac','./test_ddpg']



data_files = ['data_train_0.pkl','data_train_1.pkl','data_train_2.pkl','data_train_3.pkl','data_train_4.pkl'
              ,'data_train_5.pkl',
              #'data_test.pkl',
              #'data_trade.pkl'
              ]

price_array_path = ['price_array_train_0.pkl','price_array_train_1.pkl','price_array_train_2.pkl','price_array_train_3.pkl','price_array_train_4.pkl'
                    ,'price_array_train_5.pkl',
                    #'price_array_test.pkl'
                    #,'price_array_trade.pkl'
                    ]

tech_array_path = ['tech_array_train_0.pkl','tech_array_train_1.pkl','tech_array_train_2.pkl','tech_array_train_3.pkl','tech_array_train_4.pkl'
                     ,'tech_array_train_5.pkl',
                        #'tech_array_test.pkl'
                        #,'tech_array_trade.pkl'
                        ]

turbulence_array_path = ['turbulence_array_train_0.pkl','turbulence_array_train_1.pkl','turbulence_array_train_2.pkl','turbulence_array_train_3.pkl','turbulence_array_train_4.pkl'
                            ,'turbulence_array_train_5.pkl',
                            #'turbulence_array_test.pkl'
                            #,'turbulence_array_trade.pkl'
                            ]

trajectories = []

kwargs = ({})

pickle_path = 'pickle_collections/'

n = 20 #generate 10 trajectories for each model

for i in range(n):
    for model_name, cwd in zip(models_name, models_path):
        for data_file, price_array_file, tech_array_file, turbulence_array_file in zip(data_files, price_array_path, tech_array_path, turbulence_array_path):
            try:
                with open(data_file, "rb") as f:
                    data = pickle.load(f)
                with open(price_array_file, "rb") as f:
                    price_array = pickle.load(f)
                with open(tech_array_file, "rb") as f:
                    tech_array = pickle.load(f)
                with open(turbulence_array_file, "rb") as f:
                    turbulence_array = pickle.load(f)
            except:
                with open(pickle_path+data_file, "rb") as f:
                    data = pickle.load(f)
                with open(pickle_path+price_array_file, "rb") as f:
                    price_array = pickle.load(f)
                with open(pickle_path+tech_array_file, "rb") as f:
                    tech_array = pickle.load(f)
                with open(pickle_path+turbulence_array_file, "rb") as f:
                    turbulence_array = pickle.load(f)

            env_config = {
                "price_array": price_array,
                "tech_array": tech_array,
                "turbulence_array": turbulence_array,
                "if_train": True,
            }
            env = StockTradingEnv
            #check if 'train' in data_file
            if 'train' in data_file:
                env_instance = env(config=env_config, turbulence_thresh = 800)
            else:
                env_instance = env(config=env_config)
            episode_data = DRLAgent_sb3.get_trajectory_load_from_file(
                model_name=model_name, environment=env_instance, cwd="trained_models/"+cwd
            )
            trajectories.append(episode_data)
            
with open("trajectories.pkl", "wb") as f:
    pickle.dump(trajectories, f)
