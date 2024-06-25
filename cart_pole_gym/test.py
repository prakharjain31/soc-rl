import gym
import numpy as np
import matplotlib.pyplot as plt
granularity_0 = 50
granularity_1 = 50
granularity_2 = 50
granularity_3 = 50

env = gym.make("CartPole-v1" , render_mode="rgb_array")
s = env.reset()
# print(s[0])
state, reward, done, truncated, info = env.step(0)
# print(state)
# def choose_action(state):
#     if(np.random.random() < 0.2):
#         return env.action_space.sample()
#     else:
#         return np.argmax(Q[state[0] , state[1], state[2] , state[3] , :])

# print(choose_action(s))
# print(s)
# # print(env.observation_space.n)
# Q = np.zeros(( granularity_0 , granularity_1 , granularity_2 , granularity_3  , 2))
# Q[0,0,0,0,0] = 1
# print(Q[0,0,0,0,1])
# while True:
#     action = int(input("Action: "))
#     if action in (0, 1):
#         x = env.step(action)
#         print(f"v:{x[0][1]} , w:{x[0][3]}")
#         env.render()

plt.subplot(2, 1, 1)
plt.plot( [i for i in range(4)], [20,30,40,10] ,  label = 1)
plt.plot( [i for i in range(4)], [30,60,20,10] ,  label = 2)
plt.legend()
plt.subplot(2, 1, 2)
plt.plot( [i for i in range(4)], [10,20,30,40] ,  label = 2)
plt.plot( [i for i in range(4)], [10,10,00,90] ,  label = 2)
plt.legend()

plt.show()