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
    
load_model = False

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
        self.input_shape = (8,8,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

        if load_model:
            self.model.load_weights(model_path + "/CNN_model_agnt_0.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', \
                         padding = 'valid', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding = 'valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
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

        states      = np.zeros((self.batch_size, 8, 8, 1))
        next_states = np.zeros((self.batch_size, 8, 8, 1))
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

class DDQN_agnt_1:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # these is hyper parameters for the Double DQN
        self.learning_rate = 0.0005
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.149
        self.epsilon_min = 0.0005
        self.epsilon_decay = 0.99
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 252, 252
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        self.input_shape = (8,8,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

        if load_model:
            self.model.load_weights(model_path + "/CNN_model_agnt_1.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', \
                         padding = 'valid', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding = 'valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
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

        states      = np.zeros((self.batch_size, 8, 8, 1))
        next_states = np.zeros((self.batch_size, 8, 8, 1))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i]      = minibatch[i][0]
            actions.append(  minibatch[i][1])
            rewards.append(  minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(    minibatch[i][4])

        q_value          = self.model.predict(states)
        q_value_next     = self.model.predict(next_states)
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
                a = np.argmax(q_value_next[i])
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (tgt_q_value_next[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate *= self.epsilon_decay
            
class DDQN_agnt_2:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # these is hyper parameters for the Double DQN
        self.learning_rate = 0.0005
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.149
        self.epsilon_min = 0.0005
        self.epsilon_decay = 0.99
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 253, 253
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        self.input_shape = (8,8,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

        if load_model:
            self.model.load_weights(model_path + "/CNN_model_agnt_2.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', \
                         padding = 'valid', input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding = 'valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
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

        states      = np.zeros((self.batch_size, 8, 8, 1))
        next_states = np.zeros((self.batch_size, 8, 8, 1))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i]      = minibatch[i][0]
            actions.append(  minibatch[i][1])
            rewards.append(  minibatch[i][2])
            next_states[i] = minibatch[i][3]
            dones.append(    minibatch[i][4])

        q_value          = self.model.predict(states)
        q_value_next     = self.model.predict(next_states)
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
                a = np.argmax(q_value_next[i])
                q_value[i][actions[i]] = rewards[i] + self.discount_factor * (tgt_q_value_next[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(states, q_value, batch_size=self.batch_size, epochs=1, verbose=0)
        
        if self.epsilon_rate > self.epsilon_min:
            self.epsilon_rate *= self.epsilon_decay
            
class Dueling_agnt_3:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        # get size of state and action
        self.progress = " "
        self.action_size = action_size
        self.state_size = state_size
        
        # These are hyper parameters for the Dueling_agnt_3
        self.learning_rate = 0.0005
        self.discount_factor = 0.99
        
        self.epsilon_max = 0.149
        self.epsilon_min = 0.0005
        self.epsilon_decay = 0.99
        self.epsilon_rate = self.epsilon_max
        
        self.hidden1, self.hidden2 = 254, 254
        
        # Parameter for Experience Replay
        self.size_replay_memory = 10000
        self.batch_size = 32
        self.input_shape = (8,8,1)
        
        # Experience Replay 
        self.memory = deque(maxlen=self.size_replay_memory)
        
        # Parameter for Target Network
        self.target_update_cycle = 100

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.Copy_Weights()

        if load_model:
            self.model.load_weights(model_path + "/CNN_model_agnt_3.h5")

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

        states      = np.zeros((self.batch_size, 8, 8, 1))
        next_states = np.zeros((self.batch_size, 8, 8, 1))
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
    agnt_1 = DDQN_agnt_1(state_size, action_size)
    agnt_2 = DDQN_agnt_2(state_size, action_size)
    agnt_3 = Dueling_agnt_3(state_size, action_size)
    
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
        n_rows = 10
        n_cols = 10

        state_flags = np.zeros((8,8))
        state_agents = np.zeros((8,8))

        flag_rows = np.zeros((1,4))
        flag_cols = np.zeros((1,4))
        agent_rows = np.zeros((1,4))
        agent_cols = np.zeros((1,4))
        
        """
        len_unique = 0
        while len_unique != 8:
            init_state = np.random.randint(low=0, high=64, size=8)
            len_unique = len(np.unique(init_state))

        for idx in range(4):
            row_idx,col_idx = divmod(init_state[idx],8)
            state_flags[row_idx][col_idx] = 1
            flag_rows[0][idx] = row_idx
            flag_cols[0][idx] = col_idx

        for idx in range(4):
            row_idx,col_idx = divmod(init_state[idx+4],8)
            agent_rows[0][idx] = row_idx
            agent_cols[0][idx] = col_idx

        game_flags = np.zeros((n_rows,10))
        game_flags[1:9,1:9] = state_flags
        
        # print(flag_rows)
        # print(flag_cols)
        # print(agent_rows)
        # print(agent_cols)

        flag_rows += 1
        flag_cols += 1
        agent_rows += 1
        agent_cols += 1
        """        
        
        flag_rows  = [[1,8,1,8]]
        flag_cols  = [[1,1,8,8]]
        agent_rows = [[4,5,4,5]]
        agent_cols = [[4,4,5,5]]
        game_flags = np.zeros((n_rows,10))
        
        game_flags[int(flag_rows[0][0])][int(flag_cols[0][0])] = 1
        game_flags[int(flag_rows[0][1])][int(flag_cols[0][1])] = 1
        game_flags[int(flag_rows[0][2])][int(flag_cols[0][2])] = 1
        game_flags[int(flag_rows[0][3])][int(flag_cols[0][3])] = 1
        
        agent_0 = np.zeros((n_rows,n_cols))
        agent_1 = np.zeros((n_rows,n_cols))
        agent_2 = np.zeros((n_rows,n_cols))
        agent_3 = np.zeros((n_rows,n_cols))
        
        agent_0[int(agent_rows[0][0])][int(agent_cols[0][0])] = 2
        agent_1[int(agent_rows[0][1])][int(agent_cols[0][1])] = 2
        agent_2[int(agent_rows[0][2])][int(agent_cols[0][2])] = 2
        agent_3[int(agent_rows[0][3])][int(agent_cols[0][3])] = 2

        game_arr_frame = np.full((n_rows, n_cols), 8)
        game_arr_frame[1:9,1:9] = np.zeros((8,8))
        game_arr = game_arr_frame + game_flags + agent_0 + agent_1 + agent_2 + agent_3
        
        act_arr = np.zeros((1,4))
        
        state_t = game_arr[1:9,1:9]
        state = copy.deepcopy(state_t)
        state = state.reshape(1,8,8,1)
        
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

            ep_step += 1
            time_step += 1
            
            agnt_0_row = int(copy.deepcopy(agent_rows[0][0]))
            agnt_1_row = int(copy.deepcopy(agent_rows[0][1]))
            agnt_2_row = int(copy.deepcopy(agent_rows[0][2]))
            agnt_3_row = int(copy.deepcopy(agent_rows[0][3]))

            agnt_0_col = int(copy.deepcopy(agent_cols[0][0]))
            agnt_1_col = int(copy.deepcopy(agent_cols[0][1]))
            agnt_2_col = int(copy.deepcopy(agent_cols[0][2]))
            agnt_3_col = int(copy.deepcopy(agent_cols[0][3]))
            
            # print("agent rows :",agent_rows)
            # print("agent cols :",agent_cols)
            
            act_arr[0][0] = agnt_0.get_action(state)
            
            if game_arr[agnt_0_row][agnt_0_col] == 3:
                act_arr[0][0] = 4
            if act_arr[0][0] == 0:
                if game_arr[agnt_0_row+1][agnt_0_col] < 2:
                    agnt_0_row += 1
            if act_arr[0][0] == 1:
                if game_arr[agnt_0_row-1][agnt_0_col] < 2:
                    agnt_0_row -= 1
            if act_arr[0][0] == 2:
                if game_arr[agnt_0_row][agnt_0_col-1] < 2:
                    agnt_0_col -= 1
            if act_arr[0][0] == 3:
                if game_arr[agnt_0_row][agnt_0_col+1] < 2:
                    agnt_0_col += 1
            
            agent_0 = np.zeros((10,10))
            agent_0[agnt_0_row][agnt_0_col] = 2
            
            game_arr = game_arr_frame + game_flags + agent_0 + agent_1 + agent_2 + agent_3
            next_state_t = game_arr[1:9,1:9]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,8,8,1)
            state_1 = next_state
            
            act_arr[0][1] = agnt_1.get_action(state_1)
            if game_arr[agnt_1_row][agnt_1_col] == 3:
                act_arr[0][1] = 4
            
            if act_arr[0][1] == 0:
                if game_arr[agnt_1_row+1][agnt_1_col] < 2:
                    agnt_1_row += 1
            if act_arr[0][1] == 1:
                if game_arr[agnt_1_row-1][agnt_1_col] < 2:
                    agnt_1_row -= 1
            if act_arr[0][1] == 2:
                if game_arr[agnt_1_row][agnt_1_col-1] < 2:
                    agnt_1_col -= 1
            if act_arr[0][1] == 3:
                if game_arr[agnt_1_row][agnt_1_col+1] < 2:
                    agnt_1_col += 1

            agent_1 = np.zeros((10,10))
            agent_1[agnt_1_row][agnt_1_col] = 2
            game_arr = game_arr_frame + game_flags + agent_0 + agent_1 + agent_2 + agent_3
            next_state_t = game_arr[1:9,1:9]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,8,8,1)
            state_2 = next_state
            
            act_arr[0][2] = agnt_2.get_action(state_2)
            if game_arr[agnt_2_row][agnt_2_col] == 3:
                act_arr[0][2] = 4

            if act_arr[0][2] == 0:
                if game_arr[agnt_2_row+1][agnt_2_col] < 2:
                    agnt_2_row += 1
            if act_arr[0][2] == 1:
                if game_arr[agnt_2_row-1][agnt_2_col] < 2:
                    agnt_2_row -= 1
            if act_arr[0][2] == 2:
                if game_arr[agnt_2_row][agnt_2_col-1] < 2:
                    agnt_2_col -= 1
            if act_arr[0][2] == 3:
                if game_arr[agnt_2_row][agnt_2_col+1] < 2:
                    agnt_2_col += 1

            agent_2 = np.zeros((10,10))
            agent_2[agnt_2_row][agnt_2_col] = 2
            
            game_arr = game_arr_frame + game_flags + agent_0 + agent_1 + agent_2 + agent_3
            next_state_t = game_arr[1:9,1:9]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,8,8,1)
            state_3 = next_state
            
            act_arr[0][3] = agnt_3.get_action(state_3)
            if game_arr[agnt_3_row][agnt_3_col] == 3:
                act_arr[0][3] = 4            
            if act_arr[0][3] == 0:
                if game_arr[agnt_3_row+1][agnt_3_col] < 2:
                    agnt_3_row += 1
            if act_arr[0][3] == 1:
                if game_arr[agnt_3_row-1][agnt_3_col] < 2:
                    agnt_3_row -= 1
            if act_arr[0][3] == 2:
                if game_arr[agnt_3_row][agnt_3_col-1] < 2:
                    agnt_3_col -= 1
            if act_arr[0][3] == 3:
                if game_arr[agnt_3_row][agnt_3_col+1] < 2:
                    agnt_3_col += 1

            agent_3 = np.zeros((10,10))
            agent_3[agnt_3_row][agnt_3_col] = 2
            
            game_arr = game_arr_frame + game_flags + agent_0 + agent_1 + agent_2 + agent_3
            
            next_state_t = game_arr[1:9,1:9]
            next_state = copy.deepcopy(next_state_t)
            next_state = next_state.reshape(1,8,8,1)
        
            agent_rows[0][0] = agnt_0_row
            agent_rows[0][1] = agnt_1_row
            agent_rows[0][2] = agnt_2_row
            agent_rows[0][3] = agnt_3_row
            
            agent_cols[0][0] = agnt_0_col
            agent_cols[0][1] = agnt_1_col
            agent_cols[0][2] = agnt_2_col
            agent_cols[0][3] = agnt_3_col            
                
            distance_0 = np.zeros((1,4))
            distance_1 = np.zeros((1,4))
            distance_2 = np.zeros((1,4))
            distance_3 = np.zeros((1,4))
            
            for idx in range(4):
                if game_arr[int(flag_rows[0][idx])][int(flag_cols[0][idx])] == 1:
                    distance_0[0][idx] = np.abs(flag_rows[0][idx]-agent_rows[0][0]) \
                        + np.abs(flag_cols[0][idx]-agent_cols[0][0])
                else:
                    distance_0[0][idx] = n_rows + n_cols

            for idx in range(4):
                if game_arr[int(flag_rows[0][idx])][int(flag_cols[0][idx])] == 1:
                    distance_1[0][idx] = np.abs(flag_rows[0][idx]-agent_rows[0][1]) \
                        + np.abs(flag_cols[0][idx]-agent_cols[0][1])
                else:
                    distance_1[0][idx] = n_rows + n_cols

            for idx in range(4):
                if game_arr[int(flag_rows[0][idx])][int(flag_cols[0][idx])] == 1:
                    distance_2[0][idx] = np.abs(flag_rows[0][idx]-agent_rows[0][2]) \
                        + np.abs(flag_cols[0][idx]-agent_cols[0][2])
                else:
                    distance_2[0][idx] = n_rows + n_cols

            for idx in range(4):
                if game_arr[int(flag_rows[0][idx])][int(flag_cols[0][idx])] == 1:
                    distance_3[0][idx] = np.abs(flag_rows[0][idx]-agent_rows[0][3]) \
                        + np.abs(flag_cols[0][idx]-agent_cols[0][3])
                else:
                    distance_3[0][idx] = n_rows + n_cols
                    
            dist_fl_0 = np.zeros((1,4))
            dist_fl_1 = np.zeros((1,4))
            dist_fl_2 = np.zeros((1,4))
            dist_fl_3 = np.zeros((1,4))
            
            for idx in range(4):
                temp_dis = np.abs(flag_rows[0][0]-agent_rows[0][idx]) \
                        + np.abs(flag_cols[0][0]-agent_cols[0][idx])
                dist_fl_0[0][idx] = temp_dis

            for idx in range(4):
                temp_dis = np.abs(flag_rows[0][1]-agent_rows[0][idx]) \
                        + np.abs(flag_cols[0][1]-agent_cols[0][idx])
                dist_fl_1[0][idx] = temp_dis

            for idx in range(4):
                temp_dis = np.abs(flag_rows[0][2]-agent_rows[0][idx]) \
                        + np.abs(flag_cols[0][2]-agent_cols[0][idx])
                dist_fl_2[0][idx] = temp_dis

            for idx in range(4):
                temp_dis = np.abs(flag_rows[0][3]-agent_rows[0][idx]) \
                        + np.abs(flag_cols[0][3]-agent_cols[0][idx])
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
            if flag_n_agent == 4:
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
                
    agnt_0.model.save_weights(model_path + "/CNN_model_agnt_0.h5")
    agnt_1.model.save_weights(model_path + "/CNN_model_agnt_1.h5")
    agnt_2.model.save_weights(model_path + "/CNN_model_agnt_2.h5")
    agnt_3.model.save_weights(model_path + "/CNN_model_agnt_3.h5")
    
    e = int(time.time() - start_time)
    print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    sys.exit()
                    
if __name__ == "__main__":
    main()
