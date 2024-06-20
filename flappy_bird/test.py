import flappy_bird_gymnasium
import gymnasium
import numpy as np
import sklearn.ensemble
import random
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import sklearn.tree
from sklearn.neural_network import MLPRegressor
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam






MIN_BATCH_SIZE = 8
GAMMA = 0.99
ALPHA = 0.1
def flappy(epochs=5000, audio_on = False, render_mode = "rgb_array", score_limit=None):
    timesteps = 0
    env = gymnasium.make("FlappyBird-v0", render_mode=render_mode,audio_on=audio_on, use_lidar=False, score_limit=score_limit)
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
    model2 = sklearn.ensemble.RandomForestRegressor(n_estimators=10)
    model = model2
    # model = sklearn.tree.DecisionTreeRegressor()
    # model = MLPRegressor(hidden_layer_sizes=(64,16))
    # model = Sequential()
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(2, activation='linear'))
    # model.compile(loss='mean_squared_error', optimizer=Adam())

    
    first = True

    target = np.array([])
    # obs, _ = env.reset()
    buffer = []
    EXPLORE_RATE = 1.0
    EXPLORE_RATE_DECAY = 0.995
    MIN_EXPLORE_RATE = 0.01
    for i in range(epochs):
        total_reward = 0
        state, _ = env.reset()
        state = state.reshape(1,-1)
        while True:
            # env.render()
            timesteps += 1
            X = []
            target = []
            #epsilon-greedy action selection
            if (np.random.rand() < EXPLORE_RATE):
                action = env.action_space.sample()
                
            else:
                if(not first):
                    q_values = model.predict(state)
                    action = np.argmax(q_values[0])
                else:
                    q_values = np.zeros(2)
                    action = 0
            
            next_state, reward, done, truncated, info = env.step(action)
            # if(reward == -0.5):
            #     reward = -100
            # elif(reward == 1.0):
            #     reward = 100
            # print(reward, "reward")
            total_reward += reward
            next_state = next_state.reshape(1,-1)
            buffer.append([state, action, reward, next_state,done])
            state = next_state
            if done:
                break
            if(len(buffer) < MIN_BATCH_SIZE):
                continue
            # if(len(buffer) > 10000):
            #     buffer = buffer[300:]
            # print(buffer[0])
            batch = random.sample(buffer, MIN_BATCH_SIZE)
            # print(batch[12])
            for state2, action, reward, next_state,done in batch:
            
                # print(action, reward)
                # if(state.shape[0] == 1):
                #     state = state[0]
                # if(next_state.shape[0] == 1):
                #     next_state = next_state[0]
                
                X.append(list(state2[0]))
                
                if(first):
                    q_update = reward
                    q_values2 = np.zeros(2).reshape(1   , -1)
                    q_values2[0][action] = q_update
                else:
                    if(done):
                        q_update = reward
                    else:   
                        q_update = reward + GAMMA * np.max(model.predict(next_state)[0])
                    q_values2 = model.predict(state2)
                    # q_values2[0][action] = q_values2[0][action] + (ALPHA * (q_update - q_values2[0][action]))
                    q_values2[0][action] = q_update
                    # print(q_values2, "q_values2")
                target.append(q_values2[0])
            # for i in range(len(X)):
                # print(i , X[i] , X[i].shape  , "laall")
            X = np.array(X)
            # X = X.reshape(-1, X.shape[2])
            target = np.array(target)
            # print(target)
            if(target.shape[1] == 1):
                target = target.reshape(-1, target.shape[2])
            
            # print(X, target.shape)
            if(first):
                model2.fit(X, target)
                first = False
            else:
                model2.fit(X, target)
        EXPLORE_RATE = max(MIN_EXPLORE_RATE, EXPLORE_RATE * EXPLORE_RATE_DECAY)

            
    

        print(f"Epoch: {i}, Score: {info['score']} , Total Reward: {total_reward} , Timesteps: {timesteps}")
        if(i % 5 == 0):
            model = sklearn.base.clone(model2)
        



            # # Next action:
            # # (feed the observation to your agent here)
            # action = env.action_space.sample()

            # # Processing:
            # obs, reward, terminated, _, info = env.step(action)

            # # Checking if the player is still alive
            # if terminated:
            #     break
    tree_to_plot = model.estimators_[0]
 
    # Plot the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(tree_to_plot, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree from Random Forest")
    plt.show()
    env.close()
if __name__ == "__main__":
    flappy()