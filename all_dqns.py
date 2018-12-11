#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 09:54:24 2018
@author: ibrahim, mharding
"""
import keras, tensorflow as tf, numpy as np, gym, sys, copy, argparse
import collections, random
import matplotlib.pyplot as plt
import math 

#hyperparameters
hidden_layer1 = 32 
hidden_layer2 = 32 
hidden_layer3 = 32 
gamma_CP =  0.99
gamma_MC =  1
burn_in_MC = 10000
burn_in_CP = 10000 
mem_size = 50000
g_initial_epsilon = 0.5 
g_final_epsilon = 0.1 
g_exploration_decay_steps = 10**5 
num_episodes = 10000
minibatch_size = 32 
save_weights_num_episodes = 100
steps_update_target_network_weights = 32 
k_steps_before_minibatch = 4 
# Motivated by experimental setup, para 3 of arXiv Atari DQN paper
k_frames_between_actions = 2  

def my_metric_CP(y_true, y_pred): 
    return  keras.backend.max(y_pred)
def my_metric_MC(y_true, y_pred): 
    return  keras.backend.max(y_pred)
    
steps_beyond_done = None
def next_state_func(state, action):
        # copied from gym's cartpole.py
        gravity = 9.8
        masscart = 1.0
        masspole = 0.1
        total_mass = (masspole + masscart)
        length = 0.5 # actually half the pole's length
        polemass_length = (masspole * length)
        force_mag = 10.0
        tau = 0.02  # seconds between state updates
        # Angle at which to fail the episode
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        global steps_beyond_done
        
        x, x_dot, theta, theta_dot = state
        force = force_mag if action==1 else -force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        x  = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
        state = (x,x_dot,theta,theta_dot)
        done =  x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif steps_beyond_done is None:
            # Pole just fell!
            steps_beyond_done = 0
            reward = 1.0
        else:
            steps_beyond_done += 1
            reward = 0.0
        return np.array(state), reward, done, {}
        # copied from gym's cartpole.py

class QNetwork():

    # This class essentially defines the network architecture. 
    # The network should take in state of the world as an input, 
    # and output Q values of the actions available to the agent as the output. 

    # Define mapping from environment name to 
    # list containing [state shape, n_actions, lr]
    CP_STATE_DIMS = (4,)
    MC_STATE_DIMS = (2,)
    CP_Q_DIMS = 2
    MC_Q_DIMS = 3
    DQN_LR_CP = 0.0001
    DQN_LR_MC = 0.0001
    DOUBLE_LR_CP = 0.0001
    DOUBLE_LR_MC = 0.001
    DQN_LAYERS_MC = [
            keras.layers.Dense(hidden_layer1, input_shape=MC_STATE_DIMS,
                activation='relu'),
            keras.layers.Dense(hidden_layer2, activation='relu'),
            keras.layers.Dense(hidden_layer3, activation='relu'),
            keras.layers.Dense(MC_Q_DIMS, activation='linear')
            ]
    DQN_LAYERS_CP = [
            keras.layers.Dense(hidden_layer1, input_shape=CP_STATE_DIMS,
                activation='relu'),
            keras.layers.Dense(hidden_layer2, input_shape=CP_STATE_DIMS,
                activation='relu'),
            keras.layers.Dense(CP_Q_DIMS, activation='linear')
            ]

    ENV_INFO = {'DQN': {"CartPole-v0": [CP_STATE_DIMS, CP_Q_DIMS, 
                                        DQN_LR_CP, DQN_LAYERS_CP], 
                    "MountainCar-v0": [MC_STATE_DIMS, MC_Q_DIMS,
                                        DQN_LR_MC, DQN_LAYERS_MC]}, 
                'DOUBLE' : {"CartPole-v0": [CP_STATE_DIMS, CP_Q_DIMS,
                                        DOUBLE_LR_CP, DQN_LAYERS_CP], 
                    "MountainCar-v0": [MC_STATE_DIMS, MC_Q_DIMS,
                                        DOUBLE_LR_MC, DQN_LAYERS_MC]}}


    def __init__(self, environment_name, is_double=False):
        # DQN network is instantiated using Keras
        mtype_s = 'DQN' if not is_double else 'DOUBLE'
        state_dim, Q_dim, lr, layers = self.ENV_INFO[mtype_s][environment_name]
        self.model = keras.models.Sequential(layers)
        if environment_name == 'CartPole-v0':
            self.my_metric = my_metric_CP
        if environment_name == 'MountainCar-v0':
            self.my_metric = my_metric_MC    
        self.model.compile(metrics=[self.my_metric], loss='mse', 
                optimizer=keras.optimizers.Adam(lr=lr))
        self.reduce_lr = keras.callbacks.ReduceLROnPlateau(
                            monitor='loss', factor=0.1,
                          patience=10, min_lr=0.00000000001)
    def predict(self, state, **kwargs):
        # Return network predicted q-values for given state
        return self.model.predict(state, **kwargs)

    def fit(self, pred_values, true_values, **kwargs):
        # Fit the model we're training according to fit() API
        return self.model.fit(pred_values, true_values, callbacks=[self.reduce_lr], **kwargs )

    def set_weights(self, *args, **kwargs):
        # Set weights of model according to set_weights Keras API
        return self.model.set_weights(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        # Get weights of model according to set_weights Keras API
        return self.model.get_weights(*args, **kwargs)

    def save_model_weights(self, suffix):
        # Helper function to save your model / weights. 
        self.model.save_weights(suffix)

    def load_model(self, model_file):
		# Helper function to load an existing model.
        pass

    def load_model_weights(self,weight_file):
        # Helper funciton to load model weights. 
        self.model.load_weights(weight_file)
    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        return lr

class Dueling_QNetwork(QNetwork):
    # Define mapping from environment name to 
    # list containing [state shape, n_actions, lr]
    LR_CP = 0.0001
    STREAM_CP_HIDDEN_UNITS = 128
    STREAM_MC_HIDDEN_UNITS = 64 
    LR_MC = 0.00004
    ENV_INFO = {"CartPole-v0": [(4,), 2, LR_CP, STREAM_CP_HIDDEN_UNITS], 
                "MountainCar-v0": [(2,), 3, LR_MC, STREAM_MC_HIDDEN_UNITS]}

    def __init__(self, environment_name):
        # Define your network architecture here. It is also a good idea to define any training operations 
        # and optimizers here, initialize your variables, or alternately compile your model here.  
        state_dim, Q_dim, lr, dueling_hidden_layer = \
                self.ENV_INFO[environment_name]
        if environment_name == 'CartPole-v0':
            self.my_metric = my_metric_CP
        if environment_name == 'MountainCar-v0':
            self.my_metric = my_metric_MC
        
        inputs = keras.layers.Input(shape=state_dim)
        penult_dqn_layer = None
        if environment_name == "CartPole-v0":
            print("Using Dueling network architecture for", environment_name)
            h0_out = keras.layers.Dense(hidden_layer1, activation='relu')(inputs)
            penult_dqn_layer = h0_out 
        else:
            print("Using Dueling network architecture for", environment_name)
            h0_out = keras.layers.Dense(hidden_layer1, activation='relu')(inputs)
            h1_out = keras.layers.Dense(hidden_layer2, activation='relu')(h0_out)
            penult_dqn_layer = h1_out
        # We need to diverge the network architecture into 2 fully connected
        ## streams from the output of the h1
        # First, the state-value stream: a fully-connected layer of 128 units
        ## which is then passed through to a scalar output layer
        penult_dqn_out_vs = keras.layers.Dense(dueling_hidden_layer, activation='relu')(penult_dqn_layer)
        value_out = keras.layers.Dense(1, activation='relu')(penult_dqn_out_vs)
        # In parallel, next, the advantage-value stream: similarly a fc layer of 128 units
        ## then passed to another fc layer with output size = Q_dim
        h2_out_advs = keras.layers.Dense(dueling_hidden_layer, activation='relu')(penult_dqn_layer)
        adv_out  = keras.layers.Dense(Q_dim, activation='relu')(h2_out_advs)

        # Lastly, the output of the Dueling network is defined as a function
        ## of the two streams:
        ## Q_vals = value_out - f(adv_out)  // Using broadcasting from TF
        ## where f(adv_out) = adv_out - sample_avg(adv)
        sample_avg_adv = keras.layers.Lambda(\
                lambda l_in: keras.backend.mean(l_in, axis=1, keepdims=True))(adv_out)
        f_adv = keras.layers.Subtract()([adv_out, sample_avg_adv])
        Q_vals = keras.layers.Add()([value_out, f_adv])

        self.reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                          patience=10, min_lr=0.0000001, verbose=1)
        self.model = keras.models.Model(inputs=inputs, outputs=Q_vals)
        self.model.compile(metrics=[self.my_metric], loss='mse', 
                optimizer=keras.optimizers.RMSprop(lr=lr))

    def fit(self, pred_values, true_values, **kwargs):
        # Fit the model we're training according to fit() API
        return self.model.fit(pred_values, true_values, **kwargs )

class Replay_Memory():

    def __init__(self, burn_in, memory_size=mem_size):
        # The memory essentially stores transitions recorded from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the 
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
        # You will feed this to your model to train.
        batch = random.sample(self.memory, batch_size)
        return batch

    def append(self, transition):
        # Appends transition to the memory.     
        self.memory.append(transition)

class Deep_Agent():
    DUEL_EPS_CP_INIT = 0.5
    DUEL_EPS_CP_FINAL = 0.1
    DUEL_EPS_CP_DECAY_STEPS = 5*10**4

    DUEL_EPS_MC_INIT = 0.7
    DUEL_EPS_MC_FINAL = 0.05
    DUEL_EPS_MC_DECAY_STEPS = 1*10**6
    # In this class, we will implement functions to do the following. 
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
    #        (a) Epsilon Greedy Policy.
    #         (b) Greedy Policy. 
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.
    
    def __init__(self, environment_name, model_name, render=False, num_episodes=10000, curve_episodes=200):

        # Create an instance of the network itself, as well as the memory. 
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc. 
        self.env_name = environment_name
        self.env = gym.make(environment_name)

        # Instantiate the models
        self.is_DDQN = False
        self.skip_frames = False
        initial_epsilon = g_initial_epsilon 
        final_epsilon = g_final_epsilon 
        exploration_decay_steps = g_exploration_decay_steps 
        if model_name == "dqn" or model_name == 'ddqn':
            if model_name == "dqn": 
                print("DQN model")
            if model_name == "ddqn":
                print("DDQN model")
                self.is_DDQN = True
            self.model_name = model_name
            self.model = QNetwork(environment_name, is_double = self.is_DDQN)
            self.model_target = QNetwork(environment_name, is_double = self.is_DDQN)
        else:
            self.model_name = "dueling"
            self.model = Dueling_QNetwork(environment_name)
            self.model_target = Dueling_QNetwork(environment_name)
            if environment_name == "CartPole-v0":
                initial_epsilon = self.DUEL_EPS_CP_INIT 
                final_epsilon = self.DUEL_EPS_CP_FINAL 
                exploration_decay_steps = self.DUEL_EPS_CP_DECAY_STEPS 
            else:
                initial_epsilon = self.DUEL_EPS_MC_INIT 
                final_epsilon = self.DUEL_EPS_MC_FINAL 
                exploration_decay_steps = self.DUEL_EPS_MC_DECAY_STEPS 

            print("Dueling model")


        self.burn_in = burn_in_MC if environment_name == 'MountainCar-v0' else burn_in_CP
        self.memory = Replay_Memory(self.burn_in)
        self.gamma = gamma_MC if environment_name == 'MountainCar-v0' else gamma_CP
        self.initial_epsilon = initial_epsilon
        self.epsilon = self.initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_decay_steps = exploration_decay_steps
        self.num_episodes = num_episodes
        self.minibatch_size = minibatch_size
        self.n_steps_before_update_tgt = steps_update_target_network_weights 
        self.k_steps_before_minibatch = k_steps_before_minibatch
        self.num_of_episodes_to_update_train_and_perf_curve = 200 if not curve_episodes else curve_episodes

        self.avg_training_episodes_return=[]
        self.avg_performance_episodes_return = []
        self.avg_performance_episodes_return_2SLA = []

    def epsilon_greedy_policy(self, q_values, force_epsilon=None):
        # Creating epsilon greedy probabilities to sample from.             
        eps = None
        if force_epsilon:
            eps = force_epsilon
        else:
            # Decay epsilon, save and use
            eps = max(self.final_epsilon, self.epsilon - self.initial_epsilon/self.exploration_decay_steps)
            self.epsilon = eps
        if random.random() < eps:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(q_values)
        return action

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time. 
        action = np.argmax(q_values)
        return action

    def two_step_lookahead(self,state):
        global steps_beyond_done
        sbd = steps_beyond_done
        next_state_0, reward_0, done_0, _ = next_state_func(state, 0)
        sbd1 = steps_beyond_done        
        
        if not done_0:
            next_state_0_0, reward_0_0, done_0_0, _ = next_state_func(next_state_0, 0)
            if not done_0_0:
                next_state_0_0 = np.expand_dims(next_state_0_0,0)
                return_action_0_0 = reward_0 + reward_0_0 + max(self.model.predict(next_state_0_0)[0])
            else:
                return_action_0_0 = reward_0 + reward_0_0
                
            steps_beyond_done = sbd1   
            next_state_0_1, reward_0_1, done_0_1,_ = next_state_func(next_state_0, 1)
            if not done_0_1:
                next_state_0_1 = np.expand_dims(next_state_0_1,0)
                return_action_0_1 = reward_0 + reward_0_1 + max(self.model.predict(next_state_0_1)[0])
            else:
                return_action_0_1 = reward_0 + reward_0_1    
        else:
            return_action_0_0 = reward_0
            return_action_0_1 = reward_0
        max_return_action_0 =max(return_action_0_0,return_action_0_1)
        
        steps_beyond_done = sbd
        next_state_1, reward_1, done_1, _ = next_state_func(state, 1)
        sbd2 = steps_beyond_done 
        if not done_1:
            next_state_1_0, reward_1_0, done_1_0, _ = next_state_func(next_state_1, 0)
            if not done_1_0:
                next_state_1_0 = np.expand_dims(next_state_1_0,0)
                return_action_1_0 = reward_1 + reward_1_0 + max(self.model.predict(next_state_1_0)[0])
            else:
                return_action_1_0 = reward_1 + reward_1_0
           
            steps_beyond_done = sbd2                   
            next_state_1_1, reward_1_1, done_1_1,_ = next_state_func(next_state_1, 1)
            if not done_1_1:
                next_state_1_1 = np.expand_dims(next_state_1_1,0)
                return_action_1_1 = reward_1 + reward_1_1 + max(self.model.predict(next_state_1_1)[0])
            else:
                return_action_1_1 = reward_1 + reward_1_1    
        else:
            return_action_1_0 = reward_1
            return_action_1_1 = reward_1
        max_return_action_1 =max(return_action_1_0,return_action_1_1)

        action = 0 if max_return_action_0 >= max_return_action_1 else 1
        return action

    def fast_minibatch_update(self, replay_batch, update_tgt=False):
        # Split up axes of replay_batch buffer
        states = np.array(map(lambda bs: bs[0], replay_batch))
        actions = np.array(map(lambda bs: bs[1], replay_batch))
        rewards = np.array(map(lambda bs: bs[2], replay_batch))
        next_states = np.array(map(lambda bs: bs[3], replay_batch))
        done_flags = np.array(map(lambda bs: bs[4], replay_batch))

        # For states, next_states arrays, they have an additional dimension 
        ## due to how they are saved, so we must squeeze them down one dim
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Gather target q_values from replay batch
        batch_q_values_arr= self.model.predict(states)
        tgt_q_values = None
        if self.is_DDQN:
            selected_actions = np.argmax(self.model.predict(next_states), axis=1)
            # Outputs a vector tgt_q_values to save for actions
            tgt_q_values = rewards + self.gamma * \
                self.model_target.predict(next_states)[np.arange(len(next_states)), 
                                            selected_actions]
        else:
            # Outputs a vector tgt_q_values to save for actions
            tgt_q_values = rewards + self.gamma * \
                    np.amax(self.model_target.predict(next_states), axis=1)

        # Update q_values_mat according to computed target values
        tgt_q_values[done_flags==True] = rewards[done_flags==True]
        batch_q_values_arr[np.arange(len(states)), actions] = tgt_q_values

        history = self.model.fit(states, batch_q_values_arr,
                batch_size= self.minibatch_size, epochs=1, verbose=0)

        if update_tgt:
            # Update the target model to the current model
            self.model_target.set_weights(self.model.get_weights())
    
        return history
    def minibatch_update(self, replay_batch_states, update_tgt=False):
        batch_states = []
        batch_q_values =[]
        # Gather target q_values from replay memory states
        for state, action, reward, next_state, done in replay_batch_states:
            # q_values for actions not taken will equal predicted action values
            ## thereby only loss of action taken is used in error term 
            q_values = self.model.predict(state)[0] 
            # Use target model for estimate of q(s',a')
            tgt_q_value = None
            if self.is_DDQN:
                act = np.argmax(self.model.predict(next_state)[0])
                tgt_q_value = reward + self.gamma * self.model_target.model.predict(next_state)[0][act]
            else:
                tgt_q_value = reward + self.gamma * np.amax(self.model_target.model.predict(next_state)[0])
            if done:
                q_values[action] = reward
            else:
                q_values[action] = tgt_q_value
            batch_states.append(state[0])
            batch_q_values.append(q_values)

        return np.array(batch_states), np.array(batch_q_values)

        self.model.fit(np.array(batch_states), np.array(batch_q_values),
                batch_size= self.minibatch_size, epochs=1, verbose=0)

        if update_tgt:
            # Update the target model to the current model
            self.model_target.set_weights(self.model.get_weights())

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions. 
        print ("Burning in memory ...", self.memory.burn_in, "samples to collect.")
        state = self.env.reset()
        state = np.expand_dims(state,0)
        for i in range(self.memory.burn_in):
            action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            next_state = np.expand_dims(next_state,0)
            self.memory.append((state, action, reward, next_state, done))
            if not done:
                state = next_state
            else:
                state = self.env.reset()
                state = np.expand_dims(state,0)
        print ("Burn-in complete.")

    def step_and_update(self, state, step_number, prev_action=None):
        # Returns 2-tuple of:
        ## (action, (next_state, reward, done, info) )
        ## action is chosen action of eps greedy policy for dqn
        ## (next_state, reward, done, info) is output of env.step(action)
        action = None
        if self.skip_frames and step_number % k_frames_between_actions != 0 \
                and prev_action:
            action = prev_action   
        else:
            q_values = self.model.predict(state)[0]
            action =  self.epsilon_greedy_policy(q_values)
        next_state, reward, done, info = self.env.step(action)
        # Do a batch update after kth action taken
        update_tgt_flag = False
        update_tgt_flag = not self.is_DDQN and step_number % self.n_steps_before_update_tgt == 0
        history = None
        if step_number % self.k_steps_before_minibatch == 0:
            minibatch = self.memory.sample_batch(self.minibatch_size)
            history = self.fast_minibatch_update(minibatch, update_tgt=update_tgt_flag)

            # If DDQN, then we need to randomly swap the two QNetworks
            if self.is_DDQN: ## added by Ibrahim Fri 1:23AM 
                shuffled_models = [self.model, self.model_target]
                np.random.shuffle(shuffled_models)
                self.model, self.model_target = shuffled_models
    
        return action, (next_state, reward, done, info), history

    def train(self):
        # In this function, we will train our network. 
        # If training without experience replay_memory, then you will interact with the environment 
        # in this function, while also updating your network parameters. 

        # If you are using a replay memory, you should interact with environment here, and store these 
        # transitions to memory, while also updating your model.
        eps_counter = 1
        episodes_return = []
        num_episodes_for_performance_curve = 20
        self.avg_training_episodes_return = []
        self.avg_performance_episodes_return = []

        # Step-wise vars 
        n_steps = 0 

        # Run num_episodes many episodes
        for episode in range(self.num_episodes):
            state = self.env.reset()
            state = np.expand_dims(state,0)
            returns = 0
            done = False
            prev_action = None
            batch_lrs = []
            eps_metrics = []
            while not done:
                action, step_info, history = self.step_and_update(state, n_steps, prev_action=prev_action)
                next_state, reward, done, _ = step_info
                n_steps += 1
                next_state = np.expand_dims(next_state,0)
                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                returns += reward 
                prev_action = action
                if history:
                    #eps_metrics.append(history.history[self.model.my_metric.__name__])
                    #batch_lrs.append(history.history['lr'])
                    eps_metrics.append(history.history['loss'])
            eps_avg_metric = np.max(eps_metrics)    
            eps_avg_lr = keras.backend.get_value(self.model.model.optimizer.lr) #np.mean(batch_lrs)
            episodes_return.append(returns)
            print("Episode: ", eps_counter," Reward: ", returns, "epsilon: ", self.epsilon, "LR: ", eps_avg_lr,\
                    "Avg loss:", eps_avg_metric )#self.model.on_epoch_end(1))
            ## get the points of the training curve
            if eps_counter % self.num_of_episodes_to_update_train_and_perf_curve == 0:
                self.avg_training_episodes_return.append(sum(episodes_return)/self.num_of_episodes_to_update_train_and_perf_curve)
                episodes_return = []
            if eps_counter % save_weights_num_episodes == 0:
                filename = 'laptop_weights_'+str(self.env_name)+'_'+str(self.model_name)+'_'+str(eps_counter)
                print(filename)
                self.model.save_model_weights(filename)
            eps_counter += 1
        plt.figure()
        plt.plot(self.avg_training_episodes_return,label='training_curve')
        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(self.num_of_episodes_to_update_train_and_perf_curve) + ' episodes)')
        plt.ylabel('Average Reward per Episode')
        plt.title("Training curve: " + self.model_name + "-" + self.env_name)
        plt.legend(loc='best')
        plt.show()

    def test_stats(self, n_test_episodes, model_file=None):
        # Returns the average reward over n_test_episodes of trained model
        ## and standard deviation
        total_returns = []
        for episode in range(n_test_episodes):
            state = self.env.reset()
            state = np.expand_dims(state,0)
            total_return = 0
            done =False
            while not done:
                #self.env.render()
                q_values = self.model.predict(state)[0]
                action = self.greedy_policy(q_values)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state,0)
                state = next_state
                total_return += reward 
            total_returns.append( total_return )

        # Compute test stats    
        average_test_return = np.mean(total_returns)
        std_test_return = np.std(total_returns)
        print("Mean total reward over %s episodes: %s +/- %s" % \
                (n_test_episodes,  average_test_return, std_test_return))
        return average_test_return, std_test_return

    def performance_plot_data(self,num_episodes, model_file=None):
        episodes_return = []
        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.expand_dims(state,0)
            returns = 0
            done =False
            while not done:
                q_values = self.model.predict(state)[0]
                action =  self.epsilon_greedy_policy(q_values, 
                                        force_epsilon=0.005)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.expand_dims(next_state,0)
                state = next_state
                # no discounting
                returns += reward 
            episodes_return.append(returns)
        return (sum(episodes_return)/num_episodes)

    def performance_plot_data_2_steps_LA(self,num_episodes, model_file=None):
        episodes_return = []
        for episode in range(num_episodes):
            state = self.env.reset()
            global steps_beyond_done
            steps_beyond_done = None
            returns = 0
            done =False
            while not done:
                action =  self.two_step_lookahead(state)
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                # no discounting
                returns += reward 
            episodes_return.append(returns)
        return (sum(episodes_return)/num_episodes)
    
    def performance_curves_from_weight_files(self):
        self.avg_performance_episodes_return = []
        self.avg_performance_episodes_return_2SLA = []
        start = 200
        stop = self.num_episodes +1
        step = 200
        num_episodes_for_performance_curve = 20
        for indx in range(start, stop,step):
            filename = 'laptop_weights_'+str(self.env_name)+'_'+str(self.model_name)+'_'+str(indx)
            self.model.load_model_weights(filename)
            self.avg_performance_episodes_return.append(self.performance_plot_data(num_episodes_for_performance_curve))
#            self.avg_performance_episodes_return_2SLA.append(self.performance_plot_data_2_steps_LA(num_episodes_for_performance_curve))
            

    def plots(self):
        plt.figure()
        plt.plot(self.avg_performance_episodes_return,label='performance_curve')
#        plt.plot(self.avg_performance_episodes_return_2SLA,label='performance_curve 2 steps look_ahead')
        plt.xlabel('Training Epochs (1 epoch corresponds to '+str(self.num_of_episodes_to_update_train_and_perf_curve) + ' episodes)')
        plt.ylabel('Average Reward per Episode')
        plt.title(self.model_name.capitalize() + ': "' + self.env_name)
        plt.legend(loc='best')
        plt.show()
        
    def videos(self):
        indices=[100, 3300, 6700, 10000, 200 ] #(0/3,1/3, 2/3, and 3/3 through the training process)
        def check_eps(eps_id):
            if eps_id == 0:
                return True
        for indx in indices:
            filename = 'weights_'+str(self.env_name)+'_'+str(self.model_name)+'_'+str(indx)
            print(indx)
            self.env = gym.wrappers.Monitor(self.env, './videos/', video_callable=lambda episode_id : check_eps(episode_id), uid=str(indx),mode='evaluation',force=True if indx ==indices[0] else False,resume=False if indx ==indices[0] else True )
            self.model.load_model_weights(filename)
            for episode in range(1):
                state = agent.env.reset()
                state = np.expand_dims(state,0)
                total_return = 0
                done =False
                while not done:
                    agent.env.render()
                    q_values = agent.model.predict(state)[0]
                    action = agent.greedy_policy(q_values)
                    next_state, reward, done, info = agent.env.step(action)
                    next_state = np.expand_dims(next_state,0)
                    state = next_state
                    total_return += reward 
                print("total returns from episode: ",total_return)
            
def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, default='CartPole-v0')
    parser.add_argument('--model',dest='model_name',type=str, default="dqn")
    return parser.parse_args()

if __name__ == '__main__':

    # Setting the session to allow growth, so it doesn't allocate all GPU memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # Gather commandline args
    args = parse_arguments()
    environment_name = args.env
    model_name = args.model_name

    agent = Deep_Agent(environment_name, model_name, num_episodes=10000, curve_episodes=200)
    agent.burn_in_memory()
    agent.train()
    u, std = agent.test_stats(100)  # 6.e
    agent.performance_curves_from_weight_files()
    agent.plots()
