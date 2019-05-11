import os
import random
import time
import platform
import torch
import gym
import numpy as np

def unpack_env_obs(env_obs):
    """
    ENV --> Agent Policy Interface
    ==============================

    Gathering and Crossing are both partially-observable Markov Game. env_obs returned by Env 
    is a numpy array of dimension (num_agent, NUM_FRAME*20*10), which represents the agents' 
    observations of the game.

    The NUM_FRAME*20*10 elements (view_box) encodes NUM_FRAME layers of 10x20 pixels frames 
    in the format:
    
        (viewbox_width, viewbox_depth, NUM_FRAME).
    
    This code reshapes the above into stacked frames that can be accepted by the Policy class:
    (batch_idx, in_channel, width, height)
    
    """
    
    obs = []

    for i in range(len(env_obs)):
        x = env_obs[i]   # take the indexed agent's observation

        x = torch.Tensor(x.copy())   # Convert to tensor
        
        # Policy is a 3-layer CNN
        width, height, num_frames = x.shape
        x = x.view(1, width, height, -1)   # reshape into environment defined stacked frames
        x = x.permute(0, 3, 1, 2)  # permute to Policy accepted stacked frames
        obs.append(x)
        
    return obs  # return a list of Policy accepted stacked frames (tensor)


"""
For now, we do not implement LSTM            
# LSTM Change: Need to cycle hx and cx thru function
def select_action(model, state, lstm_hc, cuda):
    hx , cx = lstm_hc 
    num_frames, height, width = state.shape
    state = torch.FloatTensor(state.reshape(-1, num_frames, height, width))

    if cuda:
        state = state.cuda()

    probs, value, (hx, cx) = model((Variable(state), (hx, cx)))

    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)
    # LSTM Change: Need to cycle hx and cx thru function
    return action.data[0], log_prob, value, (hx, cx)
"""


def load_info(agents, params, info, narrate=False):
    """
    ENV --> Agent Policy Interface
    ==============================

    This code load info outputted by env.step() into the agents' models - their Policy.

    """

    for i in range(len(agents)):    
        agents[i].load_info(info[i])

        if narrate:
            # Only crawlers can fire laser
            if params[i]['type'] is 'crawler':
                if agents[i].tagged:
                    print('Agent{} is tagged'.format(i))
                if agents[i].laser_fired:
                    print('Agent{} fires its laser'.format(i))
                    print('and hit {} US and {} THEM'.format(agents[i].US_hit, agents[i].THEM_hit))
    return



def select_action(model, obs, cuda):
    """
    This code expects obs to be an array of stacked frames of the following dim:
    (batch_idx, in_channel, width, height)
    
    This is inputted into model - the agent's Policy, which outputs a probability 
    distribution over available actions.
    
    Policy gradient is implemented using torch.distributions.Categorical. 
    """
    
    # Policy is a 3-layer CNN
    # _, num_frames, width, height = obs.shape
    # obs = torch.FloatTensor(obs.reshape(-1, num_frames, width, height))
    
    # Policy is a 2-layer NN for now
    # obs = obs.view(1, -1)
   
    if cuda:
        obs = obs.cuda()
      
    probs = model(obs)
    m = torch.distributions.Categorical(probs)
    action = m.sample()
    log_prob = m.log_prob(action)

    return action.item(), log_prob 