import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

granularity_0 = 8
granularity_1 = 8
granularity_2 = 32
granularity_3 = 16

EPSILON_INIT = 1
EPSILON_DECAY = 0.9994
EPSILON_MIN = 0.05
EPSILON = EPSILON_INIT
DISCOUNT_FACTOR = 0.9
LEARNING_RATE = 0.9
LEARNING_RATE_DECAY = 0.99999
LAMBDA = 0.4

env = gym.make("CartPole-v1" , render_mode="rgb_array")



Q = np.zeros(( granularity_0 , granularity_1 , granularity_2 , granularity_3  , 2))

def choose_action(state):
    global EPSILON
    EPSILON = max(EPSILON_MIN , EPSILON * EPSILON_DECAY)
    if(np.random.random() < EPSILON):
        return int(env.action_space.sample())
    else:
        return np.argmax(Q[int(state[0]) , int(state[1]), int(state[2]) , int(state[3]) , :])

def discretize_state(state):
    state[0] = int(((state[0] + 4.8) / (9.6)) * granularity_0 )
    state[1] = int(((state[1] + 4.0) / (8.0)) * granularity_1 )
    state[2] = int(((state[2] + 0.42) / (0.84)) * granularity_2 )
    state[3] = int(((state[3] + 4.0) / (8.0)) * granularity_3 )
    # print(state)
    return (state)
n = 40000
mean_width = 100
reward_list = np.zeros(n+1)
average_reward_list = np.zeros(n+1)
mean_reward_list = np.zeros(n-mean_width+1)

ep_count = 0

while(n):
    n -= 1
    ep_count += 1
    E = np.zeros(( granularity_0 , granularity_1 , granularity_2 , granularity_3  , 2))
    state_prev = env.reset()[0]
    state_prev = discretize_state(state_prev)
    action_prev = env.action_space.sample()
    total_reward = 0
    while(True):
        state, reward, done, truncated, info = env.step(action_prev)
        total_reward += reward
        state = discretize_state(state)
        # print(state)
        action = choose_action(state)
        # print(state)
        Td_error = reward + (DISCOUNT_FACTOR * Q[int(state[0]) , int(state[1]), int(state[2]) , int(state[3]) , action]) - Q[int(state_prev[0]) , int(state_prev[1]), int(state_prev[2]) , int(state_prev[3]) , action_prev]
        E[int(state_prev[0]) , int(state_prev[1]), int(state_prev[2]) , int(state_prev[3]) , action_prev] += 1
        # for i in range(granularity_0):
        #     for j in range(granularity_1):
        #         for k in range(granularity_2):
        #             for l in range(granularity_3):
        #                 for m in range(2):
        #                     Q[i,j,k,l,m] += LEARNING_RATE * Td_error * E[i,j,k,l,m]
        #                     E[i,j,k,l,m] *= (DISCOUNT_FACTOR * LAMBDA)
        Q += LEARNING_RATE * Td_error * E
        E *= (DISCOUNT_FACTOR * LAMBDA)
        state_prev = state
        action_prev = action
        if done:
            break
    # LEARNING_RATE = max(0.1 , LEARNING_RATE * LEARNING_RATE_DECAY)
    reward_list[ep_count] = total_reward
    average_reward_list[ep_count] = reward_list[:ep_count].mean()
    if ep_count > mean_width:
        mean_reward_list[ep_count-mean_width] = reward_list[ep_count-mean_width:ep_count].mean()
    print(f"total_reward : {total_reward} , epsilon : {EPSILON} , episode : {ep_count}")
# print(reward_list)
# print(average_reward_list)
# plt.plot(average_reward)
# plt.show()
plt.plot(average_reward_list)
plt.show()
plt.plot(mean_reward_list)
plt.show()
         


