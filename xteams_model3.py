import os
import random
import time
import platform
import torch
import gym
import numpy as np
from collections import deque
from torch.autograd import Variable

"""
5-28-2019

(1) We will implement a Strategist class. 

A strategist analyzes the strategic position of the teams of agents it is responsible for directing, 
based on observable game space and game metrics provided by the Environment. Implemented as a black 
box, it outputs a Task/Objective for each team.

(2) We will evolve the Team class to dole out team rewards to its agents based on behavioral and 
mission awards. Each Team will have a Culture and a Mission.

A team's Culture is used to shape an agent's behavior. It does so by adding an "imaginary" behavioral 
reward/penalty on top of the reward given to the agent by the environment during training. Doing so 
shapes the agent's policy so that it conforms to the cultural norms expected by the team.

A team's Mission converts the strategist's goal into a "imaginary" mission reward, which is added 
on top of the agent's environmental and behavorial rewards during training. Doing so shape the 
agent's policy so that it achieves the goals demanded of the team by the strategist. 

A team can dole out behavioral and mission reward to its agents regardless of type or role.

"""
class Strategist():
    
    teams = []
    eyes = []  # Each team has a drone agent that serves as an eye for the strategist
    game_spaces = []
    game_metrics = []

    
    def __init__(self):
        super(Strategist, self).__init__()
        
        # Teams parameters
        self.teams = []
        self.eyes = []
        
        # Teams' game spaces  
        self.game_spaces = []
        
        # Teams' game metrics
        self.game_metrics = []
        
        
        # zone parameters

        # episode history

        return
    
    # This method accepts directorship of a team of agents, but only if the team has a drone agent
    # that can act as eye for the strategist.
    def accept(self, team):
        
        eye_found = False
        
        # Look for drone agent in team
        for agent in team.members:
            if agent.type is "drone":
                self.eyes.append(agent)  # assign agent as team eye
                eye_found = True
                break
        
        # Only accept directorship of a team if there is a team eye
        if eye_found:
            self.teams.append(team)  
        else:
            raise Exception('Cannot accept team directorship! Team {} has no drone.'.format(team.name))
            
        return
    
    # This method abdicates directorship of a team of agents
    def abdicate(self, team):
        try:
            self.teams.remove(team)
        except ValueError:
            print("Cannot abdicate team directorship! Team {} is not under strategist's direction.".format(team.name))
        return

    
    # This method generates a favorability topological map from the game space 
    def _topology(self, game_space):
        
        space = game_space.numpy()
        _,_,x,y = space.shape
        
        topology = np.zeros((x,y))
        
        # Generate favorability topology based on food units in 5x5 target zone
        for ix,iy in np.ndindex(x,y):
            topology[ix,iy] = np.sum(space[0,0,ix:ix+5, iy:iy+5])

        return topology


    # This "black box" method generates a set of goals after analyzing the game space and metrics
    def generate_goals(self, game_space):
        
        # Create a topology of favorability
        topology = self._topology(game_space)
        
        # Find the coordinate of highest favorability
        i,j = np.unravel_index(topology.argmax(), topology.shape)
        
        goals = [(i,j)]  # The goal is to move a team to the coordinate of highest favorability
        
        return goals, topology
    
        
    def _assign_goal(self, goal, team):
        
        # TBD
        
        return  
    
    # This method flush the strategist's history at the end of a game episode    
    def clear_history(self):
        
        return

    # This method resets strategist by abdicating all team directorships
    def reset(self):
        # Abdicate directorship for all teams
        self.teams = []
        self.eyes = []
        self.game_spaces = []
        self.game_metrics = []
        
        return

class Crawler_Policy(torch.nn.Module):
    """
    We implement a 3-layer convolutional network for a crawler agent
    identified by agent_inx. We comment out the LSTM implementation for now!
    """

    def __init__(self, input_channels, num_actions, agent_idx=1):
        super(Crawler_Policy, self).__init__()
        
        # Team parameters
        self.team = None
        self.color = None
        self.type = 'crawler'
        self.role = None   # Add  agent's team role  5-01-2019
        self.idx = agent_idx   # This allows multiple learning agents       
        
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0

        # zone parameters
        self.in_banned = False  # 03-02-2019
        self.in_target = False
        
        self.temperature = 1.0               # This is to adjust exploit/explore 
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        self.features = self._init_features()
        self.action_head = self._init_action_head()
        
        # Deactivate actor-critic (CNN-LSTM) for now
        # self.lstm = self._init_lstm()
        # self.action_head = self._init_action_head()
        # self.value_head = self._init_value_head()

        # episode history
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1
        self.tagged_hist = []       
        self.tag_hist = []
        self.US_hits = []
        self.THEM_hits = []
        self.in_banned_hist = []    # 03-02-2019
        self.in_target_hist = []
        

    def _init_features(self):
        
        layers = []
        
        # [1,input_channels,10,20] input 3D array
        layers.append(torch.nn.Conv2d(self.input_channels,
                                      16, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,10,20] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,5,10] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=3, stride=1, padding=0))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,3,8] feature maps

        return torch.nn.Sequential(*layers)

    def _init_action_head(self):
        # input [1,384]
        return torch.nn.Linear(384, self.num_actions)   # output [1,8]
    
    """
    # Disable CNN-LSTM actor critic for now

    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)
    """
       
    
    def forward(self, inputs):
        x = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 384(16x3x8)

        """
        # Disable CNN-LSTM actor critic for now
        
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        

        value = self.value_head(x)
        return action, value, (hx, cx)       
        
        """
        probs = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        return probs

    # This method attach the agent to a team by team name and color
    def attach_to_team(self, team_name, team_color, team_role):
        self.team = team_name
        self.color = team_color
        self.role = team_role

    # This method resets agent info 
    def reset_info(self):
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0
        self.in_banned = False  # 03-02-2019
        self.in_target = False
    
    # This method loads agent info 
    def load_info(self, info):
        # laser parameters
        self.tagged, self.laser_fired, self.US_hit, self.THEM_hit, self.in_banned, self.in_target = info
        
        # save in episode history (to be used in team reward calculation)
        self.tagged_hist.append(self.tagged)       
        self.tag_hist.append(self.laser_fired)
        self.US_hits.append(self.US_hit)
        self.THEM_hits.append(self.THEM_hit)
        self.in_banned_hist.append(self.in_banned)
        self.in_target_hist.append(self.in_target)

    # This method flush the agent's history at the end of a game episode    
    def clear_history(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.tagged_hist[:]   
        del self.tag_hist[:]
        del self.US_hits[:]
        del self.THEM_hits[:]
        del self.in_banned_hist[:]
        del self.in_target_hist[:]


class Drone_Policy(torch.nn.Module):
    """
    We implement a 3-layer CNN-LSTM for a drone follower agent. We comment out the LSTM implementation 
    for now!

    5-10-2019 Implement 2 target zone metrics which will be used to calculate leader reward:
    - apples_in_targetzone
    - US_in_targetzone

    """

    def __init__(self, input_channels, num_actions, agent_idx=1):
        super(Drone_Policy, self).__init__()
        
        # Team parameters
        self.team = None
        self.color = None
        self.type = 'drone'
        self.role = None   # Add  agent's team role  5-01-2019
        self.idx = agent_idx   # This allows multiple learning agents    
        
        self.temperature = 1.0               # This is to adjust exploit/explore 
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        self.features = self._init_features()
        self.action_head = self._init_action_head()
        
        # Deactivate actor-critic (CNN-LSTM) for now
        # self.lstm = self._init_lstm()
        # self.action_head = self._init_action_head()
        # self.value_head = self._init_value_head()

        # 5-10-2019 targetzone parameters
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone

        # episode history
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1

        # 5-09-2019 Add drone-leader metrics
        self.apples_hist = []   # apples in target zone
        self.US_hist = []   # US agents in target zone
        

    def _init_features(self):
        
        layers = []
        
        # [1,input_channels,100,60] input 3D array
        layers.append(torch.nn.Conv2d(self.input_channels,
                                      16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,50,30] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,25,15] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=3, stride=2, padding=0))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,12,7] feature maps

        return torch.nn.Sequential(*layers)

    def _init_action_head(self):
        # input [1,1344]
        return torch.nn.Linear(1344, self.num_actions)   # output [1,12]
    
    """
    # Disable CNN-LSTM actor critic for now

    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)
    """
       
    
    def forward(self, inputs):
        x = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 1344(16x12x7)

        """
        # Disable CNN-LSTM actor critic for now
        
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        

        value = self.value_head(x)
        return action, value, (hx, cx)       
        
        """
        probs = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        return probs

    # This method attach the agent to a team by team name and color
    def attach_to_team(self, team_name, team_color, team_role):
        self.team = team_name
        self.color = team_color
        self.role = team_role

    # This method resets agent info 
    def reset_info(self):

        # 5-10-2019 targetzone parameters
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone

        return
    
    # This method loads agent info 
    def load_info(self, info):

        # Get target zone metrics from info
        self.apples_in_targetzone = info
        
        # save in episode history (to be used for leader reward calculation)
        #self.apples_hist.append(self.apples_in_targetzone)       
        #self.US_hist.append(self.US_in_targetzone)
        self.apples_hist.append(self.apples_in_targetzone)
        self.US_hist.append(0)
        return


    # This method flush the agent's history at the end of a game episode    
    def clear_history(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]

        # 5-09-2019 Add drone-leader metrics
        del self.apples_hist[:]   
        del self.US_hist[:] 

"""
6-02-2019
We will implement several different drone leaders to find out which one can reliably learn a trajectory
from its current coordinate to the target coordinate specified by the strategist's goal.

- DroneLeader_FC32: a 2 layer FC-Softmax (32 hidden units) policy with the deltas (between drone and target coordinates) as input
- DroneLeader_FC64: a 2 layer FC-Softmax (64 hidden units) policy with the deltas (between drone and target coordinates) as input
- DroneLeader_CNN1: a CNN policy with a 1-frame game space of drone and goal locations as input

Note: The deltas input will be normalized.
"""

class DroneLeader_FC32(torch.nn.Module):
    """
    We implement a 2-layer FC NN for a drone leader. We comment out the LSTM implementation for now!

    6-1-2019 Implement 2-layer FC for drone leader. The input is the normalized delta between the drone's current 
    coordinate and the target coordinate (goal) demanded by the strategist.

    The policy works amazingly well at reaching the target coordinate. The policy appears to be general - it will
    will always reach the target coordinate even when we change the drone's starting coordinate or the
    target coordinate.
    """

    def __init__(self, goal_params, num_actions, agent_idx=1):
        super(DroneLeader_FC32, self).__init__()
        
        # Team parameters
        self.team = None
        self.color = None
        self.type = 'drone'
        self.role = None   # Add  agent's team role  5-01-2019
        self.idx = agent_idx   # This allows multiple learning agents    
        
        self.temperature = 1.0               # This is to adjust exploit/explore 
        self.goal_params = goal_params       # 5-31-2019 num of params in goal (demanded by strategist)
        self.num_actions = num_actions
        
        # 6-1-2019 There are two inputs
        self.features = self._init_features()   # 2nd input = goals (delta coordinates)
        self.action_head = self._init_action_head()
        
        # Deactivate actor-critic (CNN-LSTM) for now
        # self.lstm = self._init_lstm()
        # self.action_head = self._init_action_head()
        # self.value_head = self._init_value_head()

        # 5-10-2019 targetzone parameters
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone

        # episode history
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1

        # 6-02-2019 drone-leader metrics
        self.deltas = []    # drone leader's delta from target coordinate (goal) 
        

    # 6-1-2019 The input (delta coordinate) is inputted thru a 1-layer FC    
    def _init_features(self):
        
        layers = []
        
        # [1,2] input 
        layers.append(torch.nn.Linear(self.goal_params, 32))
        layers.append(torch.nn.ReLU(inplace=True))

        return torch.nn.Sequential(*layers)


    def _init_action_head(self):
        # input [1,32] 
        return torch.nn.Linear(32, self.num_actions)   # output [1,12]

    
    """
    # Disable CNN-LSTM actor critic for now

    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)
    """
       
    # 6-1-2019 This is essentially a 2-layer fully-connected NN with softmax output
    def forward(self, goals):

        x = self.features(goals)
        x = x.view(x.size(0), -1)  # 1 x 32

        """
        # Disable CNN-LSTM actor critic for now
        
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        value = self.value_head(x)
        return action, value, (hx, cx)       
        
        """
        probs = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        return probs


    # This method attach the agent to a team by team name and color
    def attach_to_team(self, team_name, team_color, team_role):
        self.team = team_name
        self.color = team_color
        self.role = team_role

    # This method resets agent info 
    def reset_info(self):
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone
        return
    
    # This method loads agent info (but do nothing with it!)
    def load_info(self, info):

        # Get target zone metrics from info
        self.apples_in_targetzone = info
        
        # save in episode history (to be used for leader reward calculation)
        # self.apples_hist.append(self.apples_in_targetzone)       
        # self.US_hist.append(self.US_in_targetzone)
        return

    # This method flush the agent's history at the end of a game episode    
    def clear_history(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]

        # 5-09-2019 Add drone-leader metrics
        del self.deltas[:] 
 

class DroneLeader_FC64(torch.nn.Module):
    """
    We implement a 2-layer FC NN for a drone leader. We comment out the LSTM implementation for now!

    6-1-2019 Implement 2-layer FC for drone leader. The input is the normalized delta between the drone's current 
    coordinate and the target coordinate (goal) demanded by the strategist.

    The policy works amazingly well at reaching the target coordinate. The policy appears to be general - it will
    will always reach the target coordinate even when we change the drone's starting coordinate or the
    target coordinate.
    """

    def __init__(self, goal_params, num_actions, agent_idx=1):
        super(DroneLeader_FC64, self).__init__()
        
        # Team parameters
        self.team = None
        self.color = None
        self.type = 'drone'
        self.role = None   # Add  agent's team role  5-01-2019
        self.idx = agent_idx   # This allows multiple learning agents    
        
        self.temperature = 1.0               # This is to adjust exploit/explore 
        self.goal_params = goal_params       # 5-31-2019 num of params in goal (demanded by strategist)
        self.num_actions = num_actions
        
        # 6-1-2019 There are two inputs
        self.features = self._init_features()   # 2nd input = goals (delta coordinates)
        self.action_head = self._init_action_head()
        
        # Deactivate actor-critic (CNN-LSTM) for now
        # self.lstm = self._init_lstm()
        # self.action_head = self._init_action_head()
        # self.value_head = self._init_value_head()

        # 5-10-2019 targetzone parameters
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone

        # episode history
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1

        # 6-02-2019 drone-leader metrics
        self.deltas = []    # drone leader's delta from target coordinate (goal) 
        

    # 6-1-2019 The input (delta coordinate) is inputted thru a 1-layer FC    
    def _init_features(self):
        
        layers = []
        
        # [1,2] input 
        layers.append(torch.nn.Linear(self.goal_params, 64))
        layers.append(torch.nn.ReLU(inplace=True))

        return torch.nn.Sequential(*layers)


    def _init_action_head(self):
        # input [1,64] 
        return torch.nn.Linear(64, self.num_actions)   # output [1,12]

    
    """
    # Disable CNN-LSTM actor critic for now

    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)
    """
       
    # 6-1-2019 This is essentially a 2-layer fully-connected NN with softmax output
    def forward(self, goals):

        x = self.features(goals)
        x = x.view(x.size(0), -1)  # 1 x 64

        """
        # Disable CNN-LSTM actor critic for now
        
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        value = self.value_head(x)
        return action, value, (hx, cx)       
        
        """
        probs = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        return probs


    # This method attach the agent to a team by team name and color
    def attach_to_team(self, team_name, team_color, team_role):
        self.team = team_name
        self.color = team_color
        self.role = team_role

    # This method resets agent info 
    def reset_info(self):
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone
        return
    
    # This method loads agent info (but do nothing with it!)
    def load_info(self, info):

        # Get target zone metrics from info
        self.apples_in_targetzone = info
        
        # save in episode history (to be used for leader reward calculation)
        #self.apples_hist.append(self.apples_in_targetzone)       
        #self.US_hist.append(self.US_in_targetzone)
        return

    # This method flush the agent's history at the end of a game episode    
    def clear_history(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]

        # 5-09-2019 Add drone-leader metrics
        del self.deltas[:]     


class DroneLeader_Advanced(torch.nn.Module):
    """
    We implement a dual-input CNN-LSTM for a drone leader. We comment out the LSTM implementation 
    for now!

    5-31-2019 Implement dual-input CNN for drone leader. The 1st input is the drone agent's complete
    observation space of the game. The 2nd input is the delta between the current coordinate of the
    drone and the target coordinate (goal) demanded by the strategist. 
    """

    def __init__(self, input_channels, goal_params, num_actions, agent_idx=1):
        super(DroneLeader_Advanced, self).__init__()
        
        # Team parameters
        self.team = None
        self.color = None
        self.type = 'drone'
        self.role = None   # Add  agent's team role  5-01-2019
        self.idx = agent_idx   # This allows multiple learning agents    
        
        self.temperature = 1.0               # This is to adjust exploit/explore 
        self.input_channels = input_channels
        self.goal_params = goal_params       # 5-31-2019 num of params in goal (demanded by strategist)
        self.num_actions = num_actions
        
        # 5-31-2019 There are two inputs
        self.features1 = self._init_features1()   # 2st input = observation space
        self.features2 = self._init_features2()   # 2nd input = goals (delta coordinates)
        self.action_head = self._init_action_head()
        
        # Deactivate actor-critic (CNN-LSTM) for now
        # self.lstm = self._init_lstm()
        # self.action_head = self._init_action_head()
        # self.value_head = self._init_value_head()

        # 5-10-2019 targetzone parameters
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone

        # episode history
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1

        # 5-09-2019 Add drone-leader metrics
        self.apples_hist = []   # apples in target zone
        self.US_hist = []   # US agents in target zone
        self.deltas = []    # 5-31-2019 drone leader's delta from target coordinate (goal) 
        

    # The 1st input (obs space of the drone agent) is inputted thru a 3-layer CNN    
    def _init_features1(self):
        
        layers = []
        
        # [1,input_channels,100,60] input 3D array
        layers.append(torch.nn.Conv2d(self.input_channels,
                                      16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,50,30] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,25,15] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=3, stride=2, padding=0))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,12,7] feature maps

        return torch.nn.Sequential(*layers)

    # 5-09-2019 The 2nd input (delta coordinate) is inputted thru a 1-layer FC    
    def _init_features2(self):
        
        layers = []
        
        # [1,2] input 
        layers.append(torch.nn.Linear(self.goal_params, 32))
        layers.append(torch.nn.ReLU(inplace=True))

        return torch.nn.Sequential(*layers)


    def _init_action_head(self):
        # input [1,1349] where 1376 = 16x12x7 + 32
        # return torch.nn.Linear(1376, self.num_actions)   # output [1,12]

        return torch.nn.Linear(32, self.num_actions)   # output [1,12]

    
    """
    # Disable CNN-LSTM actor critic for now

    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)
    """
       
    # 5-31-2019 This method concatenate the output of the 2 inputs into a FC that 
    # outputs the actions
    def forward(self, obs, goals):

        # 6-1-2019 Try simple FC first
        # x1 = self.features1(obs)
        x2 = self.features2(goals)

        # x1 = x1.view(x1.size(0), -1)  # 1 x 1344(16x12x7)
        x2 = x2.view(x2.size(0), -1)  # 1 x 32

        # x = torch.cat((x1, x2), dim=1)   # 1344 + 32 = 1376
        x = x2

        """
        # Disable CNN-LSTM actor critic for now
        
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        

        value = self.value_head(x)
        return action, value, (hx, cx)       
        
        """
        probs = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        return probs


    # This method attach the agent to a team by team name and color
    def attach_to_team(self, team_name, team_color, team_role):
        self.team = team_name
        self.color = team_color
        self.role = team_role

    # This method resets agent info 
    def reset_info(self):

        # 5-10-2019 targetzone parameters
        self.apples_in_targetzone = 0  # num of apples within a drone-leader's target zone
        self.US_in_targetzone = 0  # num of US agents within a drone-leader's target zone

        return
    
    # This method loads agent info 
    def load_info(self, info):

        # Get target zone metrics from info
        self.apples_in_targetzone = info
        
        # save in episode history (to be used for leader reward calculation)
        #self.apples_hist.append(self.apples_in_targetzone)       
        #self.US_hist.append(self.US_in_targetzone)
        self.apples_hist.append(self.apples_in_targetzone)
        self.US_hist.append(0)
        return


    # This method flush the agent's history at the end of a game episode    
    def clear_history(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]

        # 5-09-2019 Add drone-leader metrics
        del self.apples_hist[:]   
        del self.US_hist[:] 
        del self.deltas[:] 


# Just a dumb random agent
class Rdn_Policy():
            
    def __init__(self):
        super(Rdn_Policy, self).__init__()
        
        # Team parameters
        self.team = None
        self.color = None
        
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0

    def select_action(self, state):
        return random.randrange(0, 8)
    
    # This method attach the agent to a team by team name and color
    def attach_to_team(self, team_name, team_color):
        self.team = team_name
        self.color = team_color
        
    # This method loads agent info 
    def load_info(self, info):
        # laser parameters
        self.tagged, self.laser_fired, self.US_hit, self.THEM_hit, self.in_banned, self.in_target = info


"""
In this file, we will implement the Team Class:
- Agents attached to the Team have a "Us" versus "Them" mentality.
- The Team awards agents with team reward based on Culture.
"""
class Team():
    
    color = None
    name = None
    culture = None
    members = []

    # This is run when the team is created
    # 4-28-2019 Add roles to enable multi-role team
    # 5-01-2019 Team is initialized with agent_policies and agent_roles
    def __init__(self, name='Vikings', color='red', culture={'name':'individualist'}, roles = ['leader','follower'], \
        agent_policies=[], agent_roles=[]):

        self.color = color
        self.name = name
        self.culture = culture
        self.members = agent_policies
        self.roles = agent_roles

        # Used to implement exile and follower culture
        self.banned_zone = None
        self.target_zone = None
        
        # 5-01-2019 Agents are assigned team name, color and role when attached to a team
        for agent, role in zip(self.members, self.roles) :
            agent.attach_to_team(name, color, role)

        return
    
    def sum_rewards(self):
        # Add the rewards of the team members elementwise
        rewards = [0 for i in range(len(self.members[0].rewards))]
        for a in self.members:
            rewards = [rewards[j] + a.rewards[j] for j in range(len(rewards))]
        return rewards   # return the team's rewards 
    
    def sum_US_hits(self):
        # Add the num of friendly fires (fragging) by the team members elementwise
        US_hits = [0 for i in range(len(self.members[0].US_hits))]
        for a in self.members:
            US_hits = [US_hits[j] + a.US_hits[j] for j in range(len(US_hits))]
        return US_hits   # return the team's fragging incidents

    # 03-01-2019
    # These routines set the banned or target zones for the Team (only valid for certain cultures)    
    def set_banned_zone(self, banned_zone):
        # Set the zone from which agents are "banned" - receive a penalty for being there
        self.banned_zone = banned_zone

    def set_target_zone(self, target_zone):
        # Set the zone to which agents need to move towards - receive a reward for being there
        self.target_zone = target_zone

    
    def behavioral_awards(self, US_hits = None, THEM_hits = None, tag_hist=None, in_banned_hist=None, in_target_hist=None):
        culture = self.culture['name']
        if culture is 'cooperative':
            coop_factor = self.culture['coop_factor']
            rewards_to_team = self.sum_rewards() 
            num_members = len(self.members)
            awards = [r/num_members*coop_factor for r in rewards_to_team]
        elif culture is 'individualist':
            rewards_to_team = self.sum_rewards() 
            awards = [0 for r in rewards_to_team]
        elif culture is 'no_fragging':
            penalty = self.culture['penalty']
            awards = [friendly_fire * penalty for friendly_fire in US_hits]
        elif culture is 'pacifist':
            penalty = self.culture['laser_penalty']
            awards = [laser * penalty for laser in tag_hist]

        # Added 03-03-2019 exile vs follower    
        elif culture is 'pacifist_exile':
            laser_penalty = self.culture['laser_penalty']
            banish_penalty = self.culture['banish_penalty']
            awards = [(laser * laser_penalty + banish * banish_penalty)  for laser, banish in zip(tag_hist, in_banned_hist)]
        # elif culture is 'pacifist_follower':   # 5-01-2019 change to pacifist_leadfollow to distinguish from
        #                                        # from leaderless pacifist
        elif culture is 'pacifist_leadfollow':
            laser_penalty = self.culture['laser_penalty']
            target_reward = self.culture['target_reward']
            awards = [(laser * laser_penalty + target * target_reward)  for laser, target in zip(tag_hist, in_target_hist)]

        elif culture is 'warlike':
            penalty = self.culture['penalty']
            reward = self.culture['reward']
            awards = [friendly_fire * penalty + enemy_hit * reward \
                      for friendly_fire, enemy_hit in zip(US_hits,THEM_hits)]
        else:
            awards = None
        return awards

    # 6-1-2019 Implement mission reward based on an agent attaining the goal given to the team by a strategist
    def mission_awards(self, apples_hist = None, goal_deltas = None):

        # A very simple reward for a team leader is the total rewards gathered by its team
        # awards = self.sum_rewards() 
        
        # Another approach is to base the mission rewards on the number of apples in the target zone
        
        awards = [apples for apples in apples_hist]

        # The mission award is +4.0 if drone is at the target coordinate. If distance from target is 1, mission award
        # is 1/(0.25+1)=0.8. If distance from target is 2, mission award is 1/(0.25+2)=0.44.

        # awards = []
        # for delta in goal_deltas:
        #     distance_from_goal = abs(delta[0,0].numpy())*60 + abs(delta[0,1].numpy())*20

        #     awards.append(1/(0.25+distance_from_goal))

        # awards = [1.0/(1+abs(delta[0,0].numpy())+abs(delta[0,1].numpy())) for delta in goal_deltas]

        # For Debug only
        # print (awards)

        return awards

def calc_norm_deltas(goal, current):
    """
    The strategist assign a Goal to a team in the form of a target coordinate. This is the point of max favorability
    in the favorability topological map it generates from the game space.

    The method calculates the deltas between the droneleader's current coordinate from the target coordinate. The
    deltas are normalized to the game space's width and height.
    """
    target_x, target_y = goal
    current_x, current_y = current
    delta_x = (target_x - current_x)/60   # normalize
    delta_y = (target_y - current_y)/20    # normalize
    deltas = torch.Tensor([delta_x,delta_y])
    deltas = deltas.view(1, -1)
        
    # print(deltas)
    return deltas   


def calc_deltas(goal, current):
    """
    The strategist assign a Goal to a team in the form of a target coordinate. This is the point of max favorability
    in the favorability topological map it generates from the game space.

    The method calculates the deltas between the droneleader's current coordinate from the target coordinate. The
    deltas are NOT normalized to the game space's width and height.
    """
    target_x, target_y = goal
    current_x, current_y = current
    delta_x = (target_x - current_x)
    delta_y = (target_y - current_y)
    deltas = torch.Tensor([delta_x,delta_y])
    deltas = deltas.view(1, -1)
        
    # print(deltas)
    return deltas 

    
# 04-19-2019 Implemented new way to save and load checkpoints based on:
#     https://pytorch.org/tutorials/beginner/saving_loading_models.html

def save_model(file_name, ep, model, optimizer):
    """
    Save a training checkpoint whereby episode, model and optimizer parameters are saved into a dictionary 
    prior to being saved into the model file.
    
    Refer to:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """

    torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, file_name) 
    
    return
    
def load_model(agent, optimizer, model_file, device = torch.device("cuda")):
    """
    Convert a model file saved using pickle into a file where model, optimizer and epoch parameters
    are saved into a dictionary prior being saved.
    
    Refer to:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    
    The original file is assumed to be under orig_dir. The converted file is saved under conv_dir.
    """
    checkpoint = torch.load(model_file, map_location=device)
    episode = checkpoint['epoch']
    agent.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return episode


def finish_episode(teams, learners, optimizers, gamma, cuda, pre_trained=None):
    """ 
    In RL, policy gradient is calculated at the end of an episode and only then used to update
    the weights of an agent's policy.
    
    The code perform policy update on each learning agent independently. Reward for each time 
    step is stored in the list policy.rewards[] --> r(t)
    """  
    
    num_learners = len(learners)
    total_norms = [0 for i in range(num_learners)]
    policy_losses = [[] for i in range(num_learners)]
    losses = [[] for i in range(num_learners)]
    T_reward = []

    # Unless pre_trained is given, perform policy update on all agents
    if pre_trained is None:
        pre_trained = [False for i in range(num_learners)]
    else:
        if len(pre_trained) != len(learners):
            raise Exception('len(Pre_trained) does not match number of agents')

    for i in range(num_learners):
        if pre_trained[i]:
            # Skip policy update if the agent is pre-trained

            # Debug
            # print('Skip update on Agent{} type-role: {}.{}'.format(i, learners[i].type,learners[i].role))
            continue
        else:
            # The agent is a learning agent, perform policy update

            # Debug
            # print('Policy update on Agent{} type-role: {}.{}'.format(i, learners[i].type,learners[i].role))

            R = 0
            saved_actions = learners[i].saved_actions
        
            for t in teams:
                if t.name is learners[i].team:

                    # 5-10-2019 Implement team reward based on type and role for:
                    # - drone-leader
                    # - crawler-leader
                    # - crawler-follower/agent
                    if (learners[i].type is 'drone' and learners[i].role is 'leader'):   # drone-leader
                        # Debug
                        # print (learners[i].apples_hist)
                        T_reward = t.mission_awards(apples_hist = learners[i].apples_hist)

                    elif learners[i].type is 'crawler':    
                
                        # Based on team culture, calculate the team reward for the agent    
                        culture = t.culture['name']
            
                        if culture is 'cooperative':
                            T_reward = t.behavioral_awards()
                        elif culture is 'individualist':
                            T_reward = t.behavioral_awards()
                        elif culture is 'no_fragging':
                            T_reward = t.behavioral_awards(US_hits = learners[i].US_hits)
                        elif culture is 'pacifist':
                            T_reward = t.behavioral_awards(tag_hist = learners[i].tag_hist)
                        elif culture is 'pacifist_exile':
                            T_reward = t.behavioral_awards(tag_hist = learners[i].tag_hist, \
                                               in_banned_hist=learners[i].in_banned_hist)
                        # elif culture is 'pacifist_follower':    # 5-01-2019 change to pacifist_leadfollow 
                        #                                         # from leaderless pacifist
                        elif culture is 'pacifist_leadfollow':
                            T_reward = t.behavioral_awards(tag_hist = learners[i].tag_hist, \
                                               in_target_hist=learners[i].in_target_hist)
                        elif culture is 'warlike':
                            T_reward = t.behavioral_awards(US_hits = learners[i].US_hits,THEM_hits = learners[i].THEM_hits)
                        else:
                            T_reward = t.behavioral_awards()

                        if learners[i].role is 'leader':    # crawler-leader
                            leader_reward = t.mission_awards()
                            T_reward = [sum(x) for x in zip(T_reward, leader_reward)]

                    else:
                        raise Exception('Unexpected agent type-role: {}.{}'.format(learners[i].type,learners[i].role))
     
                    # For debug only
                    # print('Agent{} receives tribal award from Team{}'.format(i,t.name))
                    # print (T_reward)
                    # print (learners[i].rewards)
                
            # Do not implement actor-critic for now
            # value_losses = []
            
            rewards = deque()

            for r,T in zip(learners[i].rewards[::-1],T_reward[::-1]):
                # The agent is incentivized to cooperate by a  team bonus
                R = r + T + gamma * R
                rewards.appendleft(R)
                
            rewards = list(rewards)
            rewards = torch.Tensor(rewards)
            if cuda:
                rewards = rewards.cuda()

            # z-score rewards
            rewards = (rewards - rewards.mean()) / (1.1e-7+rewards.std())
            
            #Debug     
            #print (rewards)       
            
            """
            Do not implement actor-critic for now!!!
            for (log_prob, state_value), r in zip(saved_actions, rewards):
                reward = r - state_value.data[0]
                policy_losses.append(-log_prob * Variable(reward))
                r = torch.Tensor([r])
                if cuda:
                    r = r.cuda()
                value_losses.append(torch.nn.functional.smooth_l1_loss(state_value,
                                                                   Variable(r)))

            optimizer.zero_grad()
            loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
            loss.backward()        
            
            
            """
            for log_prob, r in zip(saved_actions, rewards):
                r = torch.Tensor([r])
                if cuda:
                    r = r.cuda()
                policy_losses[i].append(-log_prob * Variable(r))

            optimizers[i].zero_grad()
            losses[i] = torch.stack(policy_losses[i]).sum()
            losses[i].backward()
            
            # Gradient Clipping Update: prevent exploding gradient
            total_norms[i] = torch.nn.utils.clip_grad_norm_(learners[i].parameters(), 8000)
            
            optimizers[i].step()

    # clear all agent's history only at the end of episode; they are needed for team reward calc
    for i in range(num_learners):
        learners[i].clear_history()

    return total_norms