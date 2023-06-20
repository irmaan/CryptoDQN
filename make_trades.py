
import csv
import random
import matplotlib.pyplot as plt
import numpy as np

GAMMA = 0.1
EPSILON = 1
EPSILON_DECAY = .997
N_EPISODES = 1000
STARTING_BALANCE = 1000
AMOUNT_PER_TRADE = 100

csv_index = 0
random_rewards = []

#q_table, just a list of state objects.
q_table = []

last_action = 'Hold'
holding = False

class State:

    def __init__(self, data_dict):
        self.open = data_dict.get('Open')
        self.sma = data_dict.get('20SMA')
        self.bollingeru = data_dict.get('BollingerU')
        self.bollingerl = data_dict.get('BollingerL')
        self.rsi = data_dict.get('RSI')
        self.ma20 = data_dict.get('MA20')
        self.ma5 = data_dict.get('MA5')
        self.actions = {'Buy': 0, 'Hold': 0, 'Sell': 0}

        q_table.append(self)

    def get_self(self):
        return(self.sma, self.bollingeru, self.bollingerl, self.rsi, self.ma20, self.ma5)


def get_data():
    with open("data.csv", "r") as file:
        reader = csv.DictReader(file)
        data = list(reader)

        # Instantiate empty values to zero.
        for day in data:
            for lable, value in day.items():
                if not value:
                    day[lable] = 0

        return(data)


def print_data(data_list_dicts):
    print(data_list_dicts)

def get_action(current_state, policy, epsilon = 1):
    global last_action
    global holding
    probability = random.random()
    action_list = list()

    # print("Choosing action randomly")
    if last_action == 'Hold' and holding:
        action_list = ['Sell', 'Hold']

    elif last_action == 'Hold' and not holding:
        action_list = ['Buy', 'Hold']

    elif last_action == 'Buy':
        action_list = ['Hold', 'Sell']

    elif last_action == 'Sell':
        action_list = ['Hold', 'Buy']

    # Choose action randomly.
    if probability < epsilon:

        action = random.choice(action_list)
        last_action = action

    # Choosing action based on policy
    else:
        if policy[current_state] in action_list:
            action = policy[current_state]
        else:
            action = 'Hold'

    last_action = action

    if action == 'Buy':
        holding = True
    elif action == 'Sell':
        holding = False

    return action

def change_state(current_state, action, data_list_dicts):
    global csv_index
    csv_index += 1

    return State(data_list_dicts[csv_index])

def find_start_day(data_list_dicts):
    for _list in data_list_dicts:
        start_state = State(_list)
        full = all(start_state.get_self())
        if full:
            return start_state
        global csv_index
        csv_index += 1

def add_to_returns_and_policy(current_state, returns, policy):
    if current_state not in returns and current_state not in policy.keys():
        actions = current_state.actions
        policy[current_state] = random.choice(list(current_state.actions.keys()))
        for action in actions:
            returns[current_state, action] = []

def get_reward(action, current_state, trade_amount):
    global STARTING_BALANCE
    reward = 0
    if action == 'Buy':
        reward = -AMOUNT_PER_TRADE
    elif action == 'Sell':
        reward = trade_amount * float(current_state.open)
    return reward


def generate_episode(data_list_dicts, epsilon, policy, returns):
    global csv_index
    csv_index = 0
    trade_amount = 0

    # Initialize empty list for states_actions_rewards
    states_actions_rewards = list()

    # Find first day where all values are instantiated.
    start_state = find_start_day(data_list_dicts)

    # End state is the final time period of the data.
    end_state = State(data_list_dicts[-1])

    # While final state has not been reached.
    current_state = start_state
    action = get_action(current_state, epsilon) # Initialize first action

    while (current_state != end_state):
        # print(current_state.get_self())

        # Add current state to returns and policy.
        add_to_returns_and_policy(current_state, returns, policy)

        # Get trade details
        if action == 'Buy':
            trade_amount = AMOUNT_PER_TRADE / float(current_state.open)

        # Do action and take us to new state.
        reward = get_reward(action, current_state, trade_amount)
        global STARTING_BALANCE
        STARTING_BALANCE += reward

        states_actions_rewards.append((current_state, action, reward))
        current_state = change_state(current_state, action, data_list_dicts)

        if csv_index == len(data_list_dicts) - 1:
            current_state = end_state

        add_to_returns_and_policy(current_state, returns, policy)

        if action == 'Sell':
            trade_amount = 0

        action = get_action(current_state, policy, epsilon)

    # calculate the returns by working backwards from the terminal state
    G = 0
    states_actions_returns = []
    first = True
    for state, action, reward in reversed(states_actions_rewards):
        # a terminal state has a value of 0 by definition
        # this is the first state we encounter in the reversed list
        # we'll ignore its return (G) since it doesn't correspond to any move
        if first:
            first = False
        else:
            states_actions_returns.append((state, action, G))
        G = reward + GAMMA*G
    states_actions_returns.reverse() # back to the original order of states visited

    return states_actions_returns

def train_monte_carlo(data, episodes = 10000):

    epsilon = EPSILON

    policy = {}
    for state in q_table:
        policy[state] = random.choice(list(state.actions.keys()))

    # Initialize returns
    returns = {}
    for state in q_table:
        actions = state.actions
        for a in actions:
            returns[state, a] = []

    print(returns)

    # Loop for the number of episodes
    for i in range(episodes):
        print("Episode: {}".format(i))

        # Generate an episode
        print(epsilon)
        states_actions_returns = generate_episode(data, epsilon, policy, returns)
        if epsilon > .05:
            epsilon = epsilon * EPSILON_DECAY

        seen_state_action_pairs = set()
        for state, action, G in states_actions_returns:
            # check if we have already seen s
            # first-visit Monte Carlo optimization
            sa = (state, action)
            if sa not in seen_state_action_pairs:
                returns[sa].append(G)

                # the new Q[s][a] is the sample mean of all our returns for that (state, action)
                state.actions[action] = sum(returns[sa])/len(returns[sa])
                seen_state_action_pairs.add(sa)
                if len(returns[sa]) == 2:
                    returns[sa][0] = sum(returns[sa])/len(returns[sa])
                    returns[sa].pop()

        # calculate new policy pi(s) = argmax[a]{ Q(s,a) }
        for state in policy.keys():
            action = max(state.actions.values())
            policy[state] = action


        global STARTING_BALANCE
        random_rewards.append(STARTING_BALANCE)
        STARTING_BALANCE = 1000


    return policy

def test(data_list_dicts, epsilon, policy, returns):
    global csv_index
    csv_index = 0
    trade_amount = 0

    # Initialize empty list for states_actions_rewards
    states_actions_rewards = list()

    # Find first day where all values are instantiated.
    start_state = find_start_day(data_list_dicts)

    # End state is the final time period of the data.
    end_state = State(data_list_dicts[-1])

    # While final state has not been reached.
    current_state = start_state
    action = get_action(current_state, epsilon) # Initialize first action

    start = STARTING_BALANCE

    while (current_state != end_state):
        # print(current_state.get_self())

        # Add current state to returns and policy.
        add_to_returns_and_policy(current_state, returns, policy)

        # Get trade details
        if action == 'Buy':
            trade_amount = AMOUNT_PER_TRADE / float(current_state.open)

        # Do action and take us to new state.
        reward = get_reward(action, current_state, trade_amount)
        start += reward

        current_state = change_state(current_state, action, data_list_dicts)

        if csv_index == len(data_list_dicts) - 1:
            current_state = end_state

        add_to_returns_and_policy(current_state, returns, policy)

        if action == 'Sell':
            trade_amount = 0

        action = get_action(current_state, policy, epsilon)

    return start

if __name__ == "__main__":
    # Get the data into a list of dicts.
    data = get_data()

    # These variables will be in the train monte carlo bit.
    epsilon = 1

    policy = train_monte_carlo(data, N_EPISODES)

    returns = {}

    print(test(data, 0, policy, returns))


    # Plot the random one
    xpoints = np.array(range(N_EPISODES))
    ypoints = np.array(random_rewards)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Final Portfolio Value")

    plt.plot(xpoints, ypoints)
    plt.show()

    # Generate episode using data.
    # generate_episode(data, epsilon)

    # print("/////////////////////////////////////////////////////////")
    # for item in q_table:
    #     print(item.actions)

