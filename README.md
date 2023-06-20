##  A DQN Model to Predict Cryptocurrency Market

- Stock trading and cryptocurrency trading are both ways to potentially make money, with cryptocurrencies fluctuating in price like stocks. Traditional trading techniques have been successful, but this paper aims to explore reinforcement learning and deep reinforcement learning to optimize a cryptocurrency trading strategy that can outperform a human trader. 

- Steps to Implement a DQN Model for Cryptocurrency Market Prediction:

- Deep Q-Learning (DQN) is a variant of the Q-Learning algorithm that uses a deep convolutional neural network for Q-function approximation. It also utilizes mini-batches of random training data and older network parameters to estimate the Q-values of the next state. These contributions enhance the agent's ability to learn and make better decisions.

- Implement a Monte Carlo deep Q-learning algorithm to handle the large state space and numerous indicators and values associated with cryptocurrencies. This algorithm will be compared to a neural network-based algorithm for training effectiveness.

- Train the DQN model using a large amount of data from different coin histories to create a less generalized policy compared to other papers. This approach aims to tackle the challenge of a vast state space and provide a more accurate prediction of cryptocurrency market behavior. 


## Dataset:
- Look at data.csv & gemini_BTCUSD_1hr.csv