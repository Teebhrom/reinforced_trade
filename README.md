# Project Name
Reinforced Trade: Financial Trading with Deep Reinforcement Learning Method

## Description
This is the final project "Reinforced trade" code for CS492: Deep Reinforcement Learning and Game AI  course at KAIST. Some parts of this repository are modified from FinRL https://github.com/AI4Finance-Foundation/FinRL. and https://github.com/nikhilbarhate99/min-decision-transformer

## Tutorial

To get started, follow these steps:

1. Fetch the data from Yahoo Finance by running the following command:
  ```
  python create_data.py
  ```

2. Train the data using various online RL agents:
  ```
  python main.py --mode train
  ```

3. Generate the trajectory for the Transformer model:
  ```
  python split.py
  python get_trajectory.py
  ```

4. Train the Transformer model using the test data as the evaluation set:
  ```
  python load_env_for_transformer.py
  python transformer_train.py
  ```

5. Finally, obtain the result:
  ```
  python visulaize.py
  ```

Please note that this tutorial assumes you have already set up the necessary environment and dependencies given in requirement.txt file.

