

# Gym stuff
import gym
import gym_anytrading

# Stable baselines - rl stuff
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C
from stable_baselines import DQN

# Processing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# To get indicators.
from gym_anytrading.envs import StocksEnv
from finta import TA

import sys

# Import data.
df = pd.read_csv('gemini_BTCUSD_1hr.csv')

# Change Data to a Date Object.
df['Date'] = pd.to_datetime(df['Date'])

# Ensure that data is sorted correctly.
df.sort_values('Date', ascending=True, inplace=True)
df.set_index('Date', inplace=True)

# Fill out custom indicators.
BB_ROUND = 1
df['BBWIDTH'] = TA.OBV(df)
df['BBWIDTH'] = df['BBWIDTH'].round(BB_ROUND)

RSI_ROUND = 0
df['RSI'] = TA.RSI(df)
df['RSI'] = df['RSI'].round(RSI_ROUND)

OBV_ROUND = 1
df['OBV'] = TA.OBV(df)
df['OBV'] = df['OBV'].round(OBV_ROUND)

# Fill NA's with 0's.
df.fillna(0, inplace=True)

# Add signals to the environment
def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['BBWIDTH', 'RSI', 'OBV']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals

# Build Envrionment.
env2 = MyCustomEnv(df=df, window_size=12, frame_bound=(3000,3050))
print(env2.signal_features)
env_maker = lambda: env2
env = DummyVecEnv([env_maker])

# Set algorithm and train.
try:
    model = DQN.load("Bitcoin-ModelDQN")
    model.set_env(env)
    print("////////////////////////////////////////////////////////////")
    print("Loading model.")
    print("////////////////////////////////////////////////////////////\n")
except ValueError:
    print("////////////////////////////////////////////////////////////")
    print("Model does not exist. Creating new model")
    print("////////////////////////////////////////////////////////////\n")
model = DQN('MlpPolicy', env,learning_rate = .01, verbose=1)

# print("Framebound between {} and {}".format(x,y))
# model.learn(total_timesteps=200000)
model.save("Bitcoin-ModelDQN")

model = DQN.load("Bitcoin-ModelDQN")

# Evaluate.
for _ in range(1):
    env = MyCustomEnv(df=df, window_size=12, frame_bound=(3000,3100))
    print(env.max_possible_profit())
    # obs = env.reset()
    # while True:
    #     obs = obs[np.newaxis, ...]
    #     action, _states = model.predict(obs)
    #     print(action)
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         print("info", info)
    #         break

    obs = env.reset()
    while True:
        obs = obs[np.newaxis, ...]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(action, info)
        if done:
            print("info", info)
            break

plt.figure(figsize=(15,6))
plt.cla()
env.render_all()
plt.savefig('plot.png')



