import flappy_bird_gymnasium
import gymnasium
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import random
import math

class FlappyAgent:
    def __init__(self , action_space, state_space):
        # self.model = RandomForestRegressor(n_estimators=15)
        self.model = MLPRegressor(hidden_layer_sizes=(16,8), max_iter=1200)
        self.action_space = action_space
        self.state_space = state_space
        self.first = True
        self.memory = []

        self.EXPLORE_RATE = 1.0
        self.EXPLORE_RATE_DECAY = 0.995
        self.MIN_EXPLORE_RATE = 0.1
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 0.5
        self.DISCOUNT_FACTOR = 0.4
    def act(self , state):
        if(type(state) == tuple):
                    state = state[0]
        if(np.random.rand() < self.EXPLORE_RATE):
            return random.choice([0,1])
        else:
            if(self.first):
                return 0
            else:
                return (np.argmax((self.model.predict(np.array(state).reshape(1,-1)))[0]))
    def act2(self , state):
        if(type(state) == tuple):
                    state = state[0]
        x = np.random.rand()
        if(x < 0.05):
            return 1
        elif(x < 0.25):
            return 0
        else:
            if(self.first):
                return 0
            else:
                return (np.argmax((self.model.predict(np.array(state).reshape(1,-1)))[0]))
    
    def experience_replay(self , state , action , reward , state_next, done):
        transition = list([state , action , reward , state_next, done])
        self.memory.append(transition)
        if(len(self.memory) > 5000):
            self.memory.pop(0)
        if(len(self.memory) < self.BATCH_SIZE):
            return
        batch = random.sample(self.memory , self.BATCH_SIZE)
        # print(type(batch) , type(batch[0]) , type(batch[0][0]) , type(batch[0][1]) , type(batch[0][2]) , type(batch[0][3]) , type(batch[0][4]))
        for state , action , reward , state_next, done in batch:
            if self.first:
                q_update = reward
                q_values = np.zeros(2).reshape(1 , -1)
                self.first = False
                # print("First", state , action , reward , state_next , done)
            elif not done:
                # print("Not Done", state , type(state))
                if(type(state) == tuple):
                    state = state[0]
                q_update = self.model.predict(np.array(state).reshape(1,-1))[0][action]  \
                                + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * np.max(self.model.predict(np.array(state_next).reshape(1,-1))[0]) - self.model.predict(state.reshape(1,-1))[0][action])
                # q_update = reward + self.DISCOUNT_FACTOR * np.max(self.model.predict(np.array(state_next).reshape(1,-1))[0])
                q_values = self.model.predict(np.array(state).reshape(1,-1))
            else:
                q_update = reward
                q_values = self.model.predict(np.array(state).reshape(1,-1))
        
        
            q_values[0][action] = q_update
            # print("fitting" , state , q_values)
            self.model.fit(np.array(state).reshape(1,-1) , q_values)
            if(self.EXPLORE_RATE > self.MIN_EXPLORE_RATE):
                self.EXPLORE_RATE *= self.EXPLORE_RATE_DECAY
        return
    def experience_replay2(self , state , action , reward , state_next, done):
        transition = list([state , action , reward , state_next, done])
        self.memory.append(transition)
        if(len(self.memory) > 1000):
            self.memory.pop(0)
        if(len(self.memory) < self.BATCH_SIZE):
            return
        batch = random.sample(self.memory , self.BATCH_SIZE)
        target = []
        X = []
        # print(type(batch) , type(batch[0]) , type(batch[0][0]) , type(batch[0][1]) , type(batch[0][2]) , type(batch[0][3]) , type(batch[0][4]))
        for state , action , reward , state_next, done in batch:
            if self.first:
                q_update = reward
                q_values = np.zeros(2).reshape(1 , -1)
                
                # print("First", state , action , reward , state_next , done)
            elif not done:
                # print("Not Done", state , type(state))
                if(type(state) == tuple):
                    state = state[0]
                q_update = self.model.predict(np.array(state).reshape(1,-1))[0][action]  \
                                + self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * np.max(self.model.predict(np.array(state_next).reshape(1,-1))[0]) - self.model.predict(state.reshape(1,-1))[0][action])
                # q_update = reward + self.DISCOUNT_FACTOR * np.max(self.model.predict(np.array(state_next).reshape(1,-1))[0])
                q_values = self.model.predict(np.array(state).reshape(1,-1))
                # print(q_values)
            else:
                q_update = reward
                q_values = self.model.predict(np.array(state).reshape(1,-1))
        
        
            q_values[0][action] = q_update
            # print("fitting" , state , q_values)
            target.append(q_values[0])
            X.append(state)

            # self.model.fit(np.array(state).reshape(1,-1) , q_values)
        
        X = np.array(X)
        target = np.array(target)
        # print(X.shape , target.shape , "X and target")
        self.first = False
        self.model.fit(X , target)
        
        return
def test():
    env = gymnasium.make('FlappyBird-v0' , render_mode = 'rgb_array' , audio_on = True , use_lidar = False , score_limit = None)
    epochs = 0
    agent = FlappyAgent(env.action_space , env.observation_space)
    while True:
        state = env.reset()
        if(type(state) == tuple):
            state = state[0]
        total_reward = 0
        action_count = 0
        timesteps = 0
        while True:
            action = agent.act2(state)
            if(action == 1):
                action_count += 1
            state_next , reward , done , _ , info = env.step(action)
            timesteps += 1
            # if(math.isclose(reward , 1.0 , abs_tol = 1e-2)):
            #     reward = 20.0
            # elif(math.isclose(reward , -0.5 , abs_tol = 1e-2)):
            #     reward = -1.0
            #     done = True
            # elif(math.isclose(reward , 0.1 , abs_tol = 1e-2)):
            #     reward = 0
            # elif math.isclose(reward , -1.0 , abs_tol = 1e-2):
            #     reward = -1.0

            if (math.isclose(reward , -0.5 , abs_tol = 1e-2)):
                reward = -1.0
                done = True

            if(type(state_next) == tuple):
               state_next = state_next[0]
            total_reward += reward

            agent.experience_replay2(state , action , reward , state_next , done)

            state = state_next
            if done:
                break
        epochs += 1
        print("Epoch : " , epochs , "Score : " , info['score'] , "Total Reward : " , np.round(total_reward , 2), "Timesteps : " , timesteps, "Action Count : " , action_count)
    env.close()

if __name__ == '__main__':
    test()


            