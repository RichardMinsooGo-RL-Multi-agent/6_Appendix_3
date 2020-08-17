import sys
import pylab
import random
import numpy as np
import os
import time, datetime
from collections import deque
from keras.layers import *
from keras.models import Sequential,Model
import keras
from keras import backend as K_back
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D

state_size = 64
action_size = 5
n_cells = 11
n_agents = 6
model_path = "save_model/"
graph_path = "save_graph/"

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
load_model = False

class DQN_agent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # train time define
        self.training_time = 30*60
        
        self.episode = 0
        
        # These are hyper parameters for the DQN_agent
        self.learning_rate = 0.0005
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.049
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.99
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 1000
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        self.input_shape = (n_cells,n_cells,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        
        state = Input(shape=self.input_shape)        
        
        net1 = Convolution2D(32, kernel_size=(3, 3),activation='relu', \
                             padding = 'valid', input_shape=self.input_shape)(state)
        net2 = Convolution2D(64, kernel_size=(3, 3), activation='relu', padding = 'valid')(net1)
        net3 = MaxPooling2D(pool_size=(2, 2))(net2)
        net4 = Flatten()(net3)
        lay_2 = Dense(units=self.hidden2,activation='relu',kernel_initializer='he_uniform',\
                  name='hidden_layer_1')(net4)
        value_= Dense(units=1,activation='linear',kernel_initializer='he_uniform',\
                      name='Value_func')(lay_2)
        ac_activation = Dense(units=self.action_size,activation='linear',\
                              kernel_initializer='he_uniform',name='action')(lay_2)
        
        #Compute average of advantage function
        avg_ac_activation = Lambda(lambda x: K_back.mean(x,axis=1,keepdims=True))(ac_activation)
        
        #Concatenate value function to add it to the advantage function
        concat_value = Concatenate(axis=-1,name='concat_0')([value_,value_])
        concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(0))([avg_ac_activation,avg_ac_activation])

        for i in range(1,self.action_size-1):
            concat_value = Concatenate(axis=-1,name='concat_{}'.format(i))([concat_value,value_])
            concat_avg_ac = Concatenate(axis=-1,name='concat_ac_{}'.format(i))([concat_avg_ac,avg_ac_activation])

        #Subtract concatenated average advantage tensor with original advantage function
        ac_activation = Subtract()([ac_activation,concat_avg_ac])
        
        #Add the two (Value Function and modified advantage function)
        merged_layers = Add(name='final_layer')([concat_value,ac_activation])
        model = Model(inputs = state,outputs=merged_layers)
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def Copy_Weights(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        #Exploration vs Exploitation
        if np.random.rand() <= self.epsilon_rate:
            # print("Random action selected!!")
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        #in every action put in the memory
        self.memory.append((state, action, reward, next_state, done))
    
    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        
        minibatch = random.sample(self.memory, self.batch_size)

        states      = np.zeros((self.batch_size, n_cells, n_cells, 1))
        next_states = np.zeros((self.batch_size, n_cells, n_cells, 1))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i]      = minibatch[i][0]
            actions.append(  minibatch[i][1])
            rewards.append(  minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(    minibatch[i][4])

        q_value          = self.model.predict(states)
        tgt_q_value_next = self.target_model.predict(next_states)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(tgt_q_value_next[i]))
                
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate *= self.epsilon_decay
                        
def main():
    
    agents = []
    for obj in range(n_agents):
        obj = DQN_agent(state_size, action_size)
        agents.append(obj)

    if load_model:
        for idx in range(n_agents):
            agents[idx].model.load_weights(model_path + "/Model_dueling_6agt_3_{}.h5".format(idx))
        print("Weights are restored!")
    
    last_n_game_score = deque(maxlen=20)
    last_n_game_score.append(agents[0].ep_trial_step)
    avg_ep_step = np.mean(last_n_game_score)
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agents[0].episode = 0
    time_step = 0
    
    while agents[0].episode < 30:
    
    # while time.time() - start_time < agents[0].training_time and avg_ep_step > 50:
        
        done = False
        ep_step = 0
        
        game_flags = np.zeros((n_cells,n_cells))
                
        flag_rows = [0,0,n_cells-1,n_cells-1, 1,1] #,n_cells-2,n_cells-2]
        flag_cols = [0,n_cells-1,0,n_cells-1, 1,n_cells-2] #,1,n_cells-2]
        
        # print(flag_rows)
        # print(flag_cols)
        
        # sys.exit()
        # 8 agents-flags model
        game_flags[0][0] = 1
        game_flags[0][n_cells-1] = 1
        game_flags[n_cells-1][0] = 1
        game_flags[n_cells-1][n_cells-1] = 1
        
        game_flags[1][1] = 1
        game_flags[1][n_cells-2] = 1
        # game_flags[n_cells-2][1] = 1
        # game_flags[n_cells-2][n_cells-2] = 1
        # sys.exit()
        
        # 8 agents-flags model
        agent_rows = [4,5,6,4,5,6] #,4,5]
        agent_cols = [4,4,4,5,5,5] #,6,6]
        
        # print(game_flags)
        agent_pos = np.zeros((n_agents, n_cells, n_cells))
        
        for idx in range(n_agents):
            agent_pos[idx][agent_rows[idx]][agent_cols[idx]] = 2
            
        game_arr = game_flags + np.sum(agent_pos, axis = 0)
        
        game_arr_frame = np.full((n_cells+2, n_cells+2), 8)
        
        # print(game_arr)
        # sys.exit()
        
        act_arr = np.zeros((1,n_agents))[0]
        act_arr = act_arr.astype(np.int)

        state = copy.deepcopy(game_arr)
        # print(state)
        state = state.reshape(1,n_cells,n_cells,1)
        
        while not done and ep_step < agents[0].ep_trial_step:
            for idx in range(n_agents):
                if len(agents[idx].memory) < agents[idx].size_replay_memory:
                    agents[idx].progress = "Exploration"
                else :
                    agents[idx].progress = "Training"
            
            ep_step += 1
            time_step += 1
            
            tmp_states = np.zeros((n_agents, n_cells, n_cells))
            
            for idx in range(n_agents):
                act_arr[idx] = agents[idx].get_action(state)
                
                game_arr_frame[1:n_cells+1,1:n_cells+1] = game_arr
                if game_arr[agent_rows[idx]][agent_cols[idx]] == 3:
                    act_arr[idx] = 4
                if act_arr[idx] == 0:
                    if game_arr_frame[agent_rows[idx]+2][agent_cols[idx]+1] < 2:
                        agent_rows[idx] += 1
                if act_arr[idx] == 1:
                    if game_arr_frame[agent_rows[idx]][agent_cols[idx]+1] < 2:
                        agent_rows[idx] -= 1
                if act_arr[idx] == 2:
                    if game_arr_frame[agent_rows[idx]+1][agent_cols[idx]] < 2:
                        agent_cols[idx] -= 1
                if act_arr[idx] == 3:
                    if game_arr_frame[agent_rows[idx]+1][agent_cols[idx]+2] < 2:
                        agent_cols[idx] += 1

                agent_pos[idx] = np.zeros((n_cells, n_cells))
                agent_pos[idx][agent_rows[idx]][agent_cols[idx]] = 2
                game_arr = game_flags + np.sum(agent_pos, axis = 0)
                next_state = copy.deepcopy(game_arr)
                tmp_states[idx] = next_state
                next_state = next_state.reshape(1,n_cells,n_cells,1)
                state = next_state
            
            distances = np.zeros((n_agents, n_agents))
            for row_idx in range(n_agents):
                for idx in range(n_agents):
                    if game_arr[flag_rows[idx]][flag_cols[idx]] == 1:
                        distances[row_idx][idx] = np.abs(flag_rows[idx]-agent_rows[row_idx]) \
                            + np.abs(flag_cols[idx]-agent_cols[row_idx])
                    else:
                        distances[row_idx][idx] = n_cells + n_cells
            
            dist_flags = np.zeros((n_agents,n_agents)) 
            
            for row_idx in range(n_agents):
                for idx in range(n_agents):
                    if game_arr[flag_rows[row_idx]][flag_cols[row_idx]] == 1:
                        temp_dis = np.abs(flag_rows[row_idx]-agent_rows[idx]) \
                                + np.abs(flag_cols[row_idx]-agent_cols[idx])
                    else:
                        temp_dis = n_cells + n_cells
                    dist_flags[row_idx][idx] = temp_dis
                    
            game_dist = np.sum(dist_flags.min(axis=1))
                        
            remain_flags   = np.count_nonzero(next_state == 1)
            flag_n_agent   = np.count_nonzero(next_state == 3)
            
            if flag_n_agent == n_agents:
                done = True
                
            reward_arr = np.zeros((1,n_agents))[0]
            
            if done:
                reward_arr = np.zeros((1,n_agents))[0]
            else:
                for idx in range(n_agents):
                    reward_arr[idx] = -1 - np.min(distances[idx]) - remain_flags - game_dist
            
            # sys.exit()
            for idx in range(n_agents):
                next_state_t = tmp_states[idx]
                next_state_t = next_state_t.reshape(1,n_cells,n_cells,1)
                agents[idx].append_sample(state, act_arr[idx], reward_arr[idx], next_state_t, done)
                state = next_state_t
            
            if ep_step % 500 == 0:
                print("found flags :",flag_n_agent)
                print(game_arr.astype(int))
            
            state = next_state
            
            for idx in range(n_agents):
                if agents[idx].progress == "Training":
                    agents[idx].train_model()
                    if done or ep_step % agents[idx].target_update_cycle == 0:
                        # return# copy q_net --> target_net
                        agents[idx].Copy_Weights()
                    
            if done or ep_step == agents[0].ep_trial_step:
                if agents[0].progress == "Training":
                    # print(game_arr)
                    agents[0].episode += 1
                    last_n_game_score.append(ep_step)
                    avg_ep_step = np.mean(last_n_game_score)
                print("found flags :",flag_n_agent ," / Time step :", time_step)
                #print("episode finish!\n",game_arr)
                print("episode :{:>5d} / ep_step :{:>5d} / last 20 game avg :{:>4.1f}".format(agents[0].episode, ep_step, avg_ep_step))
                print("\n")
                break
                
    for idx in range(n_agents):
        agents[idx].model.save_weights(model_path + "/Model_dueling_6agt_3_{}.h5".format(idx))
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
