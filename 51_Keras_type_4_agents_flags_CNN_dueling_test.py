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
n_agents = 4
model_path = "save_model/"
graph_path = "save_graph/"

if not os.path.isdir(model_path):
    os.mkdir(model_path)

if not os.path.isdir(graph_path):
    os.mkdir(graph_path)
    
load_model = True

class DQN_agnt_0:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # train time define
        self.training_time = 30*60
        
        self.episode = 0
        
        # These are hyper parameters for the DQN_agnt_0
        self.learning_rate = 0.0005
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.149
        self.epsilon_min = 0.0005
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
    
    # DQN_agnt_0 에이전트의 생성
    agnt_0 = DQN_agnt_0(state_size, action_size)
    agnt_1 = DQN_agnt_0(state_size, action_size)
    agnt_2 = DQN_agnt_0(state_size, action_size)
    agnt_3 = DQN_agnt_0(state_size, action_size)
    
    if load_model:
        agnt_0.model.load_weights(model_path + "/Model_dueling_0.h5")
        agnt_1.model.load_weights(model_path + "/Model_dueling_1.h5")
        agnt_2.model.load_weights(model_path + "/Model_dueling_2.h5")
        agnt_3.model.load_weights(model_path + "/Model_dueling_3.h5")
    """
    agents = []
    for obj in range(n_agents):
        obj = DQN_agent(state_size, action_size)
        agents.append(obj)

    if load_model:
        for idx in range(n_agents):
            agents[idx].model.load_weights(model_path + "/Model_ddqn_{}.h5".format(idx))
        print("Weights are restored!")
    
    """
    last_n_game_score = deque(maxlen=20)
    last_n_game_score.append(agnt_0.ep_trial_step)
    avg_ep_step = np.mean(last_n_game_score)
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agnt_0.episode = 0
    time_step = 0
    
    # while agnt_0.episode < 50:
    
    while time.time() - start_time < agnt_0.training_time and avg_ep_step > 50:
        
        done = False
        ep_step = 0
        
        # state = env.reset()
        # state = np.reshape(state, [1, state_size])
        # n_cells = 10
        # n_cells = 10

        # state_flags = np.zeros((n_cells,n_cells))
        # state_agents = np.zeros((n_cells,n_cells))

        # flag_rows = np.zeros((1, n_agents))
        # flag_cols = np.zeros((1, n_agents))
        agent_rows = np.zeros((1, n_agents))
        agent_cols = np.zeros((1, n_agents))
        
        game_flags = np.zeros((n_cells,n_cells))
        
        
        flag_rows = [0,0,n_cells-1,n_cells-1]
        flag_cols = [0,n_cells-1,0,n_cells-1]
        
        # print(flag_rows)
        # print(flag_cols)
        
        # sys.exit()
        # 4 agents-flags model
        game_flags[0][0] = 1
        game_flags[0][n_cells-1] = 1
        game_flags[n_cells-1][0] = 1
        game_flags[n_cells-1][n_cells-1] = 1
        
        # sys.exit()
        # 8 agents-flags model
        """
        game_flags[0][0] = 1
        game_flags[0][n_cells-1] = 1
        game_flags[n_cells-1][0] = 1
        game_flags[n_cells-1][n_cells-1] = 1
        
        game_flags[1][1] = 1
        game_flags[1][n_cells-2] = 1
        game_flags[n_cells-2][1] = 1
        game_flags[n_cells-2][n_cells-2] = 1
        """
        
        # 4 agents-flags model
        agent_rows = [4,5,4,5]
        agent_cols = [4,4,5,5]
        
        # print(agent_rows)
        # print(agent_rows[0])
        
        # sys.exit()
        
        # 8 agents-flags model
        # agent_rows = [4,5,6,4,5,6,4,5]
        # agent_cols = [4,4,4,5,5,5,6,6]
        
        # print(game_flags)
        
        agent_0 = np.zeros((n_cells,n_cells))
        agent_1 = np.zeros((n_cells,n_cells))
        agent_2 = np.zeros((n_cells,n_cells))
        agent_3 = np.zeros((n_cells,n_cells))
        
        # agent_0[int(agent_rows[0])][int(agent_cols[0])] = 2
        # agent_1[int(agent_rows[1])][int(agent_cols[1])] = 2
        # agent_2[int(agent_rows[2])][int(agent_cols[2])] = 2
        # agent_3[int(agent_rows[3])][int(agent_cols[3])] = 2
        
        agent_0[agent_rows[0]][agent_cols[0]] = 2
        agent_1[agent_rows[1]][agent_cols[1]] = 2
        agent_2[agent_rows[2]][agent_cols[2]] = 2
        agent_3[agent_rows[3]][agent_cols[3]] = 2

        game_arr = game_flags + agent_0 + agent_1 + agent_2 + agent_3
        
        # print(game_arr)
        # sys.exit()
        
        act_arr = np.zeros((1,n_agents))
        
        state = copy.deepcopy(game_arr)
        state = state.reshape(1,n_cells,n_cells,1)
        
        # print(game_arr)
        # print("\nGame Started!!")
        # print("episode start!\n", game_arr)
        # sys.exit()
        # while ep_step < 20:
        while not done and ep_step < agnt_0.ep_trial_step:
            if len(agnt_0.memory) < agnt_0.size_replay_memory:
                agnt_0.progress = "Exploration"
            else :
                agnt_0.progress = "Training"

            if len(agnt_0.memory) < agnt_0.size_replay_memory:
                agnt_1.progress = "Exploration"
            else :
                agnt_1.progress = "Training"

            if len(agnt_0.memory) < agnt_0.size_replay_memory:
                agnt_2.progress = "Exploration"
            else :
                agnt_2.progress = "Training"

            if len(agnt_0.memory) < agnt_0.size_replay_memory:
                agnt_3.progress = "Exploration"
            else :
                agnt_3.progress = "Training"
            """
            for idx in range(n_agents):
                if len(agents[idx].memory) < agents[idx].size_replay_memory:
                    agents[idx].progress = "Exploration"
                else :
                    agents[idx].progress = "Training"
            """

            ep_step += 1
            time_step += 1
            
            # print("agent rows :",agent_rows)
            # print("agent cols :",agent_cols)
            # print(act_arr)
            # sys.exit()
            
            act_arr[0][0] = agnt_0.get_action(state)
            
            if game_arr[agent_rows[0]][agent_cols[0]] == 3:
                act_arr[0][0] = 4
            if act_arr[0][0] == 0:
                if (agent_rows[0]+1) < n_cells:
                    agent_rows[0] += 1
            if act_arr[0][0] == 1:
                if (agent_rows[0]-1) >= 0:
                    agent_rows[0] -= 1
            if act_arr[0][0] == 2:
                if (agent_cols[0]-1) >= 0:
                    agent_cols[0] -= 1
            if act_arr[0][0] == 3:
                if (agent_cols[0]+1) < n_cells:
                    agent_cols[0] += 1
            
            agent_0 = np.zeros((n_cells,n_cells))
            agent_0[agent_rows[0]][agent_cols[0]] = 2
            
            game_arr = game_flags + agent_0 + agent_1 + agent_2 + agent_3
            next_state = copy.deepcopy(game_arr)
            next_state = next_state.reshape(1,n_cells,n_cells,1)
            state_1 = next_state
                        
            act_arr[0][1] = agnt_1.get_action(state_1)
            if game_arr[agent_rows[1]][agent_cols[1]] == 3:
                act_arr[0][1] = 4            
            if act_arr[0][1] == 0:
                if (agent_rows[1]+1) < n_cells:
                    agent_rows[1] += 1
            if act_arr[0][1] == 1:
                if (agent_rows[1]-1) >= 0:
                    agent_rows[1] -= 1
            if act_arr[0][1] == 2:
                if (agent_cols[1]-1) >= 0:
                    agent_cols[1] -= 1
            if act_arr[0][1] == 3:
                if (agent_cols[1]+1) < n_cells:
                    agent_cols[1] += 1

            agent_1 = np.zeros((n_cells,n_cells))
            agent_1[agent_rows[1]][agent_cols[1]] = 2
            game_arr = game_flags + agent_0 + agent_1 + agent_2 + agent_3
            next_state = copy.deepcopy(game_arr)
            next_state = next_state.reshape(1,n_cells,n_cells,1)
            state_2 = next_state
            
            act_arr[0][2] = agnt_2.get_action(state_2)
            if game_arr[agent_rows[2]][agent_cols[2]] == 3:
                act_arr[0][2] = 4
            if act_arr[0][2] == 0:
                if (agent_rows[2]+1) < n_cells:
                    agent_rows[2] += 1
            if act_arr[0][2] == 1:
                if (agent_rows[2]-1) >= 0:
                    agent_rows[2] -= 1
            if act_arr[0][2] == 2:
                if (agent_cols[2]-1) >= 0:
                    agent_cols[2] -= 1
            if act_arr[0][2] == 3:
                if (agent_cols[2]+1) < n_cells:
                    agent_cols[2] += 1

            agent_2 = np.zeros((n_cells,n_cells))
            agent_2[agent_rows[2]][agent_cols[2]] = 2
            
            game_arr = game_flags + agent_0 + agent_1 + agent_2 + agent_3
            next_state = copy.deepcopy(game_arr)
            next_state = next_state.reshape(1,n_cells,n_cells,1)
            state_3 = next_state
            
            act_arr[0][3] = agnt_3.get_action(state_3)
            if game_arr[agent_rows[3]][agent_cols[3]] == 3:
                act_arr[0][3] = 4            
            if act_arr[0][3] == 0:
                if (agent_rows[3]+1) < n_cells:
                    agent_rows[3] += 1
            if act_arr[0][3] == 1:
                if (agent_rows[3]-1) >= 0:
                    agent_rows[3] -= 1
            if act_arr[0][3] == 2:
                if (agent_cols[3]-1) >= 0:
                    agent_cols[3] -= 1
            if act_arr[0][3] == 3:
                if (agent_cols[3]+1) < n_cells:
                    agent_cols[3] += 1

            agent_3 = np.zeros((n_cells,n_cells))
            agent_3[agent_rows[3]][agent_cols[3]] = 2
            
            game_arr = game_flags + agent_0 + agent_1 + agent_2 + agent_3
            
            next_state = copy.deepcopy(game_arr)
            next_state = next_state.reshape(1,n_cells,n_cells,1)
        
            # print(act_arr)
            # print(game_arr)
            # sys.exit()
            
            distance_0 = np.zeros((1,n_agents))
            distance_1 = np.zeros((1,n_agents))
            distance_2 = np.zeros((1,n_agents))
            distance_3 = np.zeros((1,n_agents))
            
            for idx in range(n_agents):
                if game_arr[flag_rows[idx]][flag_cols[idx]] == 1:
                    distance_0[0][idx] = np.abs(flag_rows[idx]-agent_rows[0]) \
                        + np.abs(flag_cols[idx]-agent_cols[0])
                else:
                    distance_0[0][idx] = n_cells + n_cells

            for idx in range(n_agents):
                if game_arr[flag_rows[idx]][flag_cols[idx]] == 1:
                    distance_1[0][idx] = np.abs(flag_rows[idx]-agent_rows[1]) \
                        + np.abs(flag_cols[idx]-agent_cols[1])
                else:
                    distance_1[0][idx] = n_cells + n_cells

            for idx in range(n_agents):
                if game_arr[flag_rows[idx]][flag_cols[idx]] == 1:
                    distance_2[0][idx] = np.abs(flag_rows[idx]-agent_rows[2]) \
                        + np.abs(flag_cols[idx]-agent_cols[2])
                else:
                    distance_2[0][idx] = n_cells + n_cells

            for idx in range(n_agents):
                if game_arr[flag_rows[idx]][flag_cols[idx]] == 1:
                    distance_3[0][idx] = np.abs(flag_rows[idx]-agent_rows[3]) \
                        + np.abs(flag_cols[idx]-agent_cols[3])
                else:
                    distance_3[0][idx] = n_cells + n_cells
                    
            dist_fl_0 = np.zeros((1,4))
            dist_fl_1 = np.zeros((1,4))
            dist_fl_2 = np.zeros((1,4))
            dist_fl_3 = np.zeros((1,4))
            
            for idx in range(n_agents):
                temp_dis = np.abs(flag_rows[0]-agent_rows[idx]) \
                        + np.abs(flag_cols[0]-agent_cols[idx])
                dist_fl_0[0][idx] = temp_dis

            for idx in range(n_agents):
                temp_dis = np.abs(flag_rows[1]-agent_rows[idx]) \
                        + np.abs(flag_cols[1]-agent_cols[idx])
                dist_fl_1[0][idx] = temp_dis

            for idx in range(n_agents):
                temp_dis = np.abs(flag_rows[2]-agent_rows[idx]) \
                        + np.abs(flag_cols[2]-agent_cols[idx])
                dist_fl_2[0][idx] = temp_dis

            for idx in range(n_agents):
                temp_dis = np.abs(flag_rows[3]-agent_rows[idx]) \
                        + np.abs(flag_cols[3]-agent_cols[idx])
                dist_fl_3[0][idx] = temp_dis
            
            game_dist = np.min(dist_fl_0) + np.min(dist_fl_1) + np.min(dist_fl_2) + np.min(dist_fl_3)
            
            remain_flags   = np.count_nonzero(next_state == 1)
            flag_n_agent   = np.count_nonzero(next_state == 3)
            """
            print("Actions    :",act_arr)
            
            print("agent rows :",agent_rows)
            print("agent cols :",agent_cols)
            
            print("Game Status :\n", game_arr,"\n")
            if ep_step == 5:
                sys.exit()
            """
            if flag_n_agent == n_agents:
                done = True
                
            if done:
                reward_0 = 0
                reward_1 = 0
                reward_2 = 0
                reward_3 = 0
            else:
                reward_0 = -1 - np.min(distance_0) - remain_flags - game_dist
                reward_1 = -1 - np.min(distance_1) - remain_flags - game_dist
                reward_2 = -1 - np.min(distance_2) - remain_flags - game_dist
                reward_3 = -1 - np.min(distance_3) - remain_flags - game_dist
            
            agnt_0.append_sample(state, int(act_arr[0][0]), reward_0, state_1, done)
            agnt_1.append_sample(state_1, int(act_arr[0][1]), reward_1, state_2, done)
            agnt_2.append_sample(state_2, int(act_arr[0][2]), reward_2, state_3, done)
            agnt_3.append_sample(state_3, int(act_arr[0][3]), reward_3, next_state, done)
            
            # agnt_0.append_sample(state, int(act_arr[0][0]), reward_0, next_state, done)
            # agnt_1.append_sample(state_1, int(act_arr[0][1]), reward_1, next_state, done)
            # agnt_2.append_sample(state_2, int(act_arr[0][2]), reward_2, next_state, done)
            
            
            # if flag_n_agent > 3:
            #     print(game_arr)
            if ep_step % 500 == 0:
                print("\nfound flags :",flag_n_agent)
                print(game_arr)
            
            state = next_state
            
            if agnt_0.progress == "Training":
                agnt_0.train_model()
                if done or ep_step % agnt_0.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agnt_0.Copy_Weights()

            if agnt_1.progress == "Training":
                agnt_1.train_model()
                if done or ep_step % agnt_1.target_update_cycle == 0:
                    agnt_1.Copy_Weights()
                    
            if agnt_2.progress == "Training":
                agnt_2.train_model()
                if done or ep_step % agnt_2.target_update_cycle == 0:
                    agnt_2.Copy_Weights()
                    
            if agnt_3.progress == "Training":
                agnt_3.train_model()
                if done or ep_step % agnt_3.target_update_cycle == 0:
                    agnt_3.Copy_Weights()
                    
            if done or ep_step == agnt_0.ep_trial_step:
                if agnt_0.progress == "Training":
                    # print(game_arr)
                    agnt_0.episode += 1
                    last_n_game_score.append(ep_step)
                    avg_ep_step = np.mean(last_n_game_score)
                print("found flags :",flag_n_agent ," / Time step :", time_step)
                #print("episode finish!\n",game_arr)
                print("episode :{:>5d} / ep_step :{:>5d} / last 20 game avg :{:>4.1f}".format(agnt_0.episode, ep_step, avg_ep_step))
                print("\n")
                break
                
    agnt_0.model.save_weights(model_path + "/Model_dueling_0.h5")
    agnt_1.model.save_weights(model_path + "/Model_dueling_1.h5")
    agnt_2.model.save_weights(model_path + "/Model_dueling_2.h5")
    agnt_3.model.save_weights(model_path + "/Model_dueling_3.h5")
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
