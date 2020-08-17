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
        self.training_time = 60*60
        
        self.episode = 0
        
        # These are hyper parameters for the DQN_agnt_0
        self.learning_rate = 0.001
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.149
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 251, 251
        
        self.ep_trial_step = 1000
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        
        self.input_shape = (11,11,1)
        
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

        states      = np.zeros((self.batch_size, 11, 11, 1))
        next_states = np.zeros((self.batch_size, 11, 11, 1))
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
            # Q Learning: get maximum Q value at s' from target model
            if dones[i]:
                q_value[i][actions[i]] = rewards[i]
            else:
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
        agnt_0.model.load_weights(model_path + "/Preda_prey_dueling_0.h5")
        agnt_1.model.load_weights(model_path + "/Preda_prey_dueling_1.h5")
        agnt_2.model.load_weights(model_path + "/Preda_prey_dueling_2.h5")
        agnt_3.model.load_weights(model_path + "/Preda_prey_dueling_3.h5")

    last_n_game_score = deque(maxlen=20)
    last_n_game_score.append(agnt_0.ep_trial_step)
    avg_ep_step = np.mean(last_n_game_score)
    
    display_time = datetime.datetime.now()
    print("\n\n Game start at :",display_time)
    
    start_time = time.time()
    agnt_0.episode = 0
    time_step = 0
    
    # while agnt_0.episode < 50:
    
    while time.time() - start_time < agnt_0.training_time and avg_ep_step > 100:
        
        done = False
        ep_step = 0
        
        n_rows = 13
        n_cols = 13

        state_prey = np.zeros((11,11))
        state_predators = np.zeros((11,11))

        prey_row = 0
        prey_col = 0
        predator_rows = np.zeros((1,4))
        predator_cols = np.zeros((1,4))

        len_unique = 0
        while len_unique != 5:
            init_state = np.random.randint(low=0, high=11*11, size=5)
            len_unique = len(np.unique(init_state))
                        
        prey_row,prey_col = divmod(init_state[0],11)
        
        state_prey[prey_row][prey_col] = 1
        
        for idx in range(4):
            row_idx,col_idx = divmod(init_state[idx+1],11)
            predator_rows[0][idx] = row_idx
            predator_cols[0][idx] = col_idx

        game_prey = np.zeros((n_rows,n_cols))
        game_prey[1:12,1:12] = state_prey

        predator_0 = np.zeros((n_rows,n_cols))
        predator_1 = np.zeros((n_rows,n_cols))
        predator_2 = np.zeros((n_rows,n_cols))
        predator_3 = np.zeros((n_rows,n_cols))

        prey_row += 1
        prey_col += 1
        predator_rows += 1
        predator_cols += 1
        
        predator_0[int(predator_rows[0][0])][int(predator_cols[0][0])] = 2
        predator_1[int(predator_rows[0][1])][int(predator_cols[0][1])] = 2
        predator_2[int(predator_rows[0][2])][int(predator_cols[0][2])] = 2
        predator_3[int(predator_rows[0][3])][int(predator_cols[0][3])] = 2

        game_arr_frame = np.full((n_rows, n_cols), 8)
        game_arr_frame[1:12,1:12] = np.zeros((11,11))
        game_arr = game_arr_frame + game_prey + predator_0 + predator_1 + predator_2 + predator_3
        
        act_arr = np.zeros((1,4))
        
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

            ep_step += 1
            time_step += 1
            
            # prey action define
            prey_action = random.randrange(5)
            if prey_action == 0:
                if game_arr[prey_row+1][prey_col] == 0:
                    prey_row += 1
            if prey_action == 1:
                if game_arr[prey_row-1][prey_col] == 0:
                    prey_row -= 1
            if prey_action == 2:
                if game_arr[prey_row][prey_col-1] == 0:
                    prey_col -= 1
            if prey_action == 3:
                if game_arr[prey_row][prey_col+1] == 0:
                    prey_col += 1
            # print("Prey Action :",prey_action)
            
            game_prey = np.zeros((13,13))
            game_prey[prey_row][prey_col] = 1
            game_arr = game_arr_frame + game_prey + predator_0 + predator_1 + predator_2 + predator_3
            # print(game_arr)
            
            state_t = game_arr[1:12,1:12]
            state = copy.deepcopy(state_t)
            state = state.reshape(1,11,11,1)
        
            # agent action define
            agnt_0_row = int(predator_rows[0][0])
            agnt_1_row = int(predator_rows[0][1])
            agnt_2_row = int(predator_rows[0][2])
            agnt_3_row = int(predator_rows[0][3])

            agnt_0_col = int(predator_cols[0][0])
            agnt_1_col = int(predator_cols[0][1])
            agnt_2_col = int(predator_cols[0][2])
            agnt_3_col = int(predator_cols[0][3])
            
            act_arr[0][0] = agnt_0.get_action(state)
            act_arr[0][1] = agnt_1.get_action(state)
            act_arr[0][2] = agnt_2.get_action(state)
            act_arr[0][3] = agnt_3.get_action(state)
            
            # Assume that below action is correct
            for idx in range(4):
                if act_arr[0][idx] == 0:
                    if game_arr[int(predator_rows[0][idx]+1)][int(predator_cols[0][idx])] == 0:
                        predator_rows[0][idx] += 1
                if act_arr[0][idx] == 1:
                    if game_arr[int(predator_rows[0][idx]-1)][int(predator_cols[0][idx])] == 0:
                        predator_rows[0][idx] -= 1
                if act_arr[0][idx] == 2:
                    if game_arr[int(predator_rows[0][idx])][int(predator_cols[0][idx]-1)] == 0:
                        predator_cols[0][idx] -= 1
                if act_arr[0][idx] == 3:
                    if game_arr[int(predator_rows[0][idx])][int(predator_cols[0][idx]+1)] == 0:
                        predator_cols[0][idx] += 1
                                                
            predator_0 = np.zeros((13,13))
            predator_1 = np.zeros((13,13))
            predator_2 = np.zeros((13,13))
            predator_3 = np.zeros((13,13))            
            predator_0[int(predator_rows[0][0])][int(predator_cols[0][0])] = 2
            predator_1[int(predator_rows[0][1])][int(predator_cols[0][1])] = 2
            predator_2[int(predator_rows[0][2])][int(predator_cols[0][2])] = 2
            predator_3[int(predator_rows[0][3])][int(predator_cols[0][3])] = 2
            
            game_arr = game_arr_frame + game_prey + predator_0 + predator_1 + predator_2 + predator_3
            
            next_state_t = game_arr[1:12,1:12]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,11,11,1)
        
            distances = np.zeros((1,4))
            
            for idx in range(4):
                temp_dis = np.abs(predator_rows[0][idx] - prey_row) \
                        + np.abs(prey_col-predator_cols[0][idx])
                distances[0][idx] = temp_dis
            
            if game_arr[prey_row+1][prey_col] != 0:
                if game_arr[prey_row-1][prey_col] != 0:
                    if game_arr[prey_row][prey_col-1] != 0:
                        if game_arr[prey_row][prey_col+1] != 0:
                            done = True                
            
            if done:
                reward_0 = 0
                reward_1 = 0
                reward_2 = 0
                reward_3 = 0
            else:
                reward_0 = -1 - np.sum(distances) - distances[0][0] + 4
                reward_1 = -1 - np.sum(distances) - distances[0][1] + 4
                reward_2 = -1 - np.sum(distances) - distances[0][2] + 4
                reward_3 = -1 - np.sum(distances) - distances[0][3] + 4
            
            agnt_0.append_sample(state, int(act_arr[0][0]), reward_0, next_state, done)
            agnt_1.append_sample(state, int(act_arr[0][1]), reward_1, next_state, done)
            agnt_2.append_sample(state, int(act_arr[0][2]), reward_2, next_state, done)
            agnt_3.append_sample(state, int(act_arr[0][3]), reward_3, next_state, done)
            
            state = next_state
            
            if agnt_0.progress == "Training":
                agnt_0.train_model()
                if done or ep_step % agnt_0.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agnt_0.Copy_Weights()

            if agnt_1.progress == "Training":
                agnt_1.train_model()
                if done or ep_step % agnt_1.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agnt_1.Copy_Weights()
                    
            if agnt_2.progress == "Training":
                agnt_2.train_model()
                if done or ep_step % agnt_2.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agnt_2.Copy_Weights()
                    
            if agnt_3.progress == "Training":
                agnt_3.train_model()
                if done or ep_step % agnt_3.target_update_cycle == 0:
                    # return# copy q_net --> target_net
                    agnt_3.Copy_Weights()
                    
            if done or ep_step == agnt_0.ep_trial_step:
                if agnt_0.progress == "Training":
                    # print(game_arr)
                    agnt_0.episode += 1
                    last_n_game_score.append(ep_step)
                    avg_ep_step = np.mean(last_n_game_score)
                print("episode :{:>5d} / ep_step :{:>5d} / last 20 game avg :{:>4.1f}".format(agnt_0.episode, ep_step, avg_ep_step))
                break
                
    agnt_0.model.save_weights(model_path + "/Preda_prey_dueling_0.h5")
    agnt_1.model.save_weights(model_path + "/Preda_prey_dueling_1.h5")
    agnt_2.model.save_weights(model_path + "/Preda_prey_dueling_2.h5")
    agnt_3.model.save_weights(model_path + "/Preda_prey_dueling_3.h5")
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
