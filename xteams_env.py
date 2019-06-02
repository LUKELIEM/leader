import collections
import itertools
import os.path
import tkinter as tk

import gym
import gym.envs.registration
import gym.spaces

import numpy as np

"""
04-20-2019
CROSSING is an environment where the agents are organized by teams with a "Us" versus "Them" mentality.
Crossing accept teams with multiple roles - leader and agents. 

There are 2 types of agents with different action and observation spaces:
- land-based Crawlers
- air-based Drones

Thus there are 4 possible type-role pairs:
- drone-leader
- drone-agent (not implemented for now)
- crawler-leader
- crawler-agent

The Crossing env has a new terrain called "river" to separate 2 food piles (one on each side). 
A land-based agent (such as a crawler) gets a -1.0 penalty for each game step it is in the river. An air-based
agent (such as a drone) is not penalized when hovering over the river.
"""

"""
CRAWLER AGENTS
"""

# Rewards and penalties
REWARD_APPLE = 1    # reward of +1 for gathering an apple
PENALTY_RIVER = -1   # penalty of -1 for each game step in the river

# Partial observation space dimension
CRAWLER_VIEW_WIDTH = 10
CRAWLER_VIEW_DEPTH = 20

# Laser beam dimension
CRAWLER_LASER_WIDTH = 3
CRAWLER_LASER_DEPTH = 10

# For Crawlers, Crossing is a partial observable Markov game. A crawler knows whether the other crawlers in its 
# observation space is US versus THEM. The observation space provided by CrossingEnv contain a stack of 7 frames 
# of 10x20 pixels:
# 1. Location of Food
# 2. Location of US agents in the viewbox
# 3. Location of THEM agents in the viewbox
# 4. Location of the walls
# 5. Location of the rivers
# 6. Location of banned zone   (added to aid exploration and lead-follow)  03-01-2019
# 7. Location of target zone   (added to aid exploration and lead-follow)  03-01-2019
NUM_FRAMES = 7

# Action Space (8 actions)
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
ROTATE_RIGHT = 4
ROTATE_LEFT = 5
LASER = 6
NOOP = 7

NUM_ACTIONS_CRAWLER = 8


"""
DRONE AGENTS
A Drone agents can observe the entire game space. It does not have (or need) an orientation. It moves N, E, S, 
W at a slow or fast pace. It can sample observations and select action at a lower frequency than the Crawlers.
It cannot gather apples and can fly over rivers, so receives no reward or penalty from either.
"""

# Action Space (12 actions)
N_SLOW = 0
E_SLOW = 1
S_SLOW = 2
W_SLOW = 3
N_FAST = 4
E_FAST = 5
S_FAST = 6
W_FAST = 7
DRONE_NOOP = 8 
# reserve for more actions later
DRONE_NOOP1 = 9 
DRONE_NOOP2 = 10
DRONE_NOOP3 = 11

NUM_ACTIONS_DRONE = 12

# Sampling Frequency
FREQ = 5        # Leader samples observations and selects action every 5 game steps.


"""
LEADER ROLE
If an agent (drone or crawler) has a Leader role within a team, the Environment sets its team's target_zone based on 
the agent's location. During rendering, the target zone is represented by a green box.
"""

# Dim of target zone set by leader 4-29-2019
TARGET_WIDTH = 5
TARGET_HEIGHT = 5



class CrossingEnv(gym.Env):

    # Some basic parameters for the Crossing Game
    metadata = {'render.modes': ['human']}
    scale = 10           # Used to scale to display during rendering

    # Viewbox is used to implement partial observable spaces for Crawler agents
    viewbox_width = CRAWLER_VIEW_WIDTH 
    viewbox_depth = CRAWLER_VIEW_DEPTH 
    padding = max(viewbox_width // 2, viewbox_depth - 1)  # essentially 20-1=19

    # To help agents distinquish between themselves, the other agents and apples
    agent_colors = []    # input during __init__()

    # A function to build the game space from a text file
    def _text_to_map(self, text):

        m = [list(row) for row in text.splitlines()]  # regard "\r", "\n", and "\r\n" as line boundaries 
        l = len(m[0])
        for row in m:   # Check for errors in text file
            if len(row) != l:
                raise ValueError('the rows in the map are not all the same length')
        
        def pad(a):
        # The function adds a padding of 20 zeros around a game space 
            return np.pad(a, self.padding + 1, 'constant')

        a = np.array(m).T   # convert to numpy array

        # Create zero-padded game spaces 
        # For example, if the map defined by the text file is 10x20, and padding is 20
        # The game space is 30x40. 
        # In the lines of codes below, game spaces for food, wall and river are created

        self.initial_food = pad(a == 'O').astype(np.int)    # Positions of food units are marked with '1'
                                                            # every other positions are marked '0'
        self.walls = pad(a == '#').astype(np.int)           # the position of wall is marked with a '1' 
                                                            # every other positions are marked '0'
        self.rivers = pad(a == 'R').astype(np.int)          # the position of river is marked with a '1' 
                                                            # every other positions are marked '0'


    # __init__() is run when the environment is created.
    # 04-23-2019
    # The env will accept the parameters of agents and teams as a list
    # 04-28-2019
    # Reorganize agents and teams parameters
    # 05-06-2019 Implement drone agents

    def __init__(self, 
        agents=[{'id': 0, 'team': 'Vikings', 'color': 'blue', 'type': 'crawler', 'role': 'follower', 'start': (1,1)}],
        teams=[{'name': 'Vikings', 'color': 'blue', 
            'culture': {'name':'pacifist','laser_penalty':-1.0},
            'roles': ['leader','follower'],
            'target_zone': None, 'banned_zone': None}], 
        map_name='default', river_penalty=0, debug_window = False, debug_agent = 0):    

        self.root = None                   # For rendering
        self.debug_window = debug_window   # 5-31-2019 Implement self.debug_window flag

        # Environment Parameters
        # ======================

        # Create game space from text file
        if not os.path.exists(map_name):
            expanded = os.path.join('maps', map_name + '.txt')
            if not os.path.exists(expanded):
                raise ValueError('map not found: ' + map_name)
            map_name = expanded
        with open(map_name) as f:
            self._text_to_map(f.read().strip())    # This sets up self.initial_food and self.walls

        # Populate the rest of environment parameters
        self.gamespace_width = self.initial_food.shape[0]
        self.gamespace_height = self.initial_food.shape[1]
        self.river_penalty = river_penalty  # penalty per time step for an agent to stay in the river

        # Team Parameters
        # ===============
        self.teams = [team['name'] for team in teams]
        self.teams_roles = [team['roles'] for team in teams]  # available roles for teams  4-29-2019

        # Banned Zones - Each team will have a banned zone, from which its agents are banned.
        # Target Zones - Each team will have a target zone, into which it desires its agents to move into.
        #
        # A zone is a rectangle defined by a tuple of tuples ((x,y),(width,height)), where (x,y) is
        # the location of the upper-left corner of the rectangle.
        self.target_zones = [team['target_zone'] for team in teams]  # target zones we want agents to move into (each team has 1)
        self.banned_zones = [team['banned_zone'] for team in teams]  # banned zones we want agents to move out of (each team has 1)


        # Agent Parameters
        # ================
        self.n_agents = len(agents)    # Set number of agents
        self.debug_agent = debug_agent  # agent for debug purpose (rendered as a separate box)

        # Agent's color, team, team role and type
        self.agent_colors = [agent['color'] for agent in agents]
        self.agent_teams = [agent['team'] for agent in agents]
        self.agent_roles = [agent['role'] for agent in agents]   # 4-29-2019 Add role to agent parameter

        #5-07-2019 Error-check agent roles
        for i, role in enumerate(self.agent_roles):
            team_index = self._find_team(i)    # find its team
            if team_index is not None:            
                if role not in self.teams_roles[team_index]:     # Check that agent's role is valid for its team
                    raise Exception('Unexpected agent role: {}'.format(role))

        self.agent_types = [agent['type'] for agent in agents]   # 4-29-2019 Add type to agent parameter

        self.spawn_points = []
        self.agent_locations = []
        for agent in agents:
            spawn_x, spawn_y = agent['start']
            self.spawn_points.append((spawn_x+self.padding, spawn_y+self.padding))
            self.agent_locations.append((spawn_x+self.padding, spawn_y+self.padding))

        # Configure spaces
        self.action_space = []
        self.observation_space = []

        # State size for the two agent types
        self.crawler_state_size = self.viewbox_width * self.viewbox_depth * NUM_FRAMES 
        self.drone_state_size = self.gamespace_width * self.gamespace_height * NUM_FRAMES   
        
        # 4-29-2019
        # Set up action and obs spaces for agents based on their types. There are 2 agents types:
        # 
        # Crawler - land-based, partial observation space of 10x20
        # Dronw - air-based, complete observation space (same as game space which is map-dependent)
        for agent in agents:
            if agent['type'] is 'crawler':     
                self.action_space.append(gym.spaces.MultiDiscrete([[0, NUM_ACTIONS_CRAWLER]]))
                self.observation_space.append(gym.spaces.MultiDiscrete([[[0, 1]] * self.crawler_state_size]))
            elif agent['type'] is 'drone': 
                # For now, implement leader role like follower role
                self.action_space.append(gym.spaces.MultiDiscrete([[0, NUM_ACTIONS_DRONE]]))
                self.observation_space.append(gym.spaces.MultiDiscrete([[[0, 1]] * self.drone_state_size]))
            else:
                raise Exception('Unexpected agent type: {}'.format(agent['type']))

        self._spec = gym.envs.registration.EnvSpec(**_spec)
        self.reset()    # Reset environment
        self.done = False


    # A function that returns the index of the team for a specific agent
    def _find_team(self, agent_index):

        for team_index, team in enumerate(self.teams):  
            if self.agent_teams[agent_index] == team:  # find its team
                return team_index

        return None   # Return None if there is no match


    # A function to check if the location the crawler agent intends to move to will result in a collision with
    # another crawler agent. 
    def _collide(self, agent_index, next_location, current_locations):

        for j, current in enumerate(current_locations):
            if (j is agent_index or                 # Skip its own current location 
                self.agent_types[j] is 'drone'):    # Skip drones  5-06-2019
                continue
            if next_location == current:   # If the location is occupied
                # print("Collide!")
                return True
        return False

    # A function that returns how many agents of same team vs different teams the agent has fired on
    # 5-06-2019 Drone cannot be hit by laser from crawlers
    def _laser_hits(self, kill_zone, agent_firing):
        US = self.agent_teams[agent_firing]   # US is the team of the agent that fires the laser
        US_hit = 0
        THEM_hit = 0   

         # In case the agent lands on a cell with food, or is tagged
        for i, a in enumerate(self.agent_locations):
            if (i is agent_firing or                  # Do not count the firing agent
                self.agent_types[i] is 'drone'):      # Skip drones 5-06-2019   
                continue
            if kill_zone[a]:
                if self.agent_teams[i] is US:
                    US_hit += 1
                else:
                    THEM_hit += 1
        return US_hit, THEM_hit

    # A function to take the game one step forward
    # Inputs: a list of actions indexed by agent
    def _step(self, action_n):

        """
        (1) Calculate proposed movements
        (2) Move the agents
        (3) Set target and banned zones and generate metrics
        (4) Implement laser tagging and generate metrics
        (5) Implement apple consumption and regeneration
        (6) Implement crawler respawning
        (7) Organize _step() return data
         """

        assert len(action_n) == self.n_agents  # Error check for action list


        # (1) Calculate proposed movements
        # Go through the list of agents and process their actions into proposed movements.

        movement_n = [(0, 0) for a in action_n]  # Initialize variables for movement

        # Debug agent movements
        # print (self.agent_locations)

        for i, agent in enumerate(self.agent_locations):
            if (self.agent_roles[i] is 'leader' and self.agent_types[i] is 'drone'):    # drone-leader

                # Process drone-leader actions [N_SLOW, E_SLOW, S_SLOW, W_SLOW, N_FAST, E_FAST, S_FAST, W_FAST]
                # Note that unlike a crawler, a drone's movement action is objective. 

                if action_n[i] in [N_SLOW, N_FAST]:
                     movement_n[i] = (0, -1)  # North
                elif action_n[i] in [E_SLOW, E_FAST]:
                     movement_n[i] = (1, 0)  # EAST
                elif action_n[i] in [S_SLOW, S_FAST]:
                     movement_n[i] = (0, 1)  # South
                elif action_n[i] in [W_SLOW, W_FAST]:
                     movement_n[i] = (-1, 0)  # West

            elif self.agent_types[i] is 'crawler':    # crawler-leader or follower

                # Process crawler actions [UP, DOWN, RIGHT, LEFT, ROTATE_RIGHT, ROTATE_LEFT]
                # A crawler's action is subjective, based on its orientation (which direction it is facing) 
                # in the game space.
                
                if self.tagged[i]:
                    action_n[i] = NOOP      # Reset action of tagged crawler agent to NOOP

                if action_n[i] in [UP, DOWN, LEFT, RIGHT]:
                # The calculation translates an agents's orientation and subjective action to x-y coordinate
                # movements in the game space    
                    a = action_n[i]
                    a = (a + self.orientations[i]) % 4
                    movement_n[i] = [
                        (0, -1),  # up/forward
                        (1, 0),   # right
                        (0, 1),   # down/backward
                        (-1, 0),  # left
                    ][a]
                elif action_n[i] is ROTATE_RIGHT:
                    self.orientations[i] = (self.orientations[i] + 1) % 4
                elif action_n[i] is ROTATE_LEFT:
                    self.orientations[i] = (self.orientations[i] - 1) % 4

            else:
                raise Exception('Invalid type-role for agent {}: {}-{}'.format(i, self.agent_types[i],self.agent_roles[i]))

        # Debug agent movements
        # print (movement_n)

        # (2) Move the agents
        # The code below updates agents' locations based on proposed movements 

        current_locations = [a for a in self.agent_locations]   # This list keeps track of where agents have moved to

        for i, ((dx, dy), (x, y)) in enumerate(zip(movement_n, self.agent_locations)):  # For each agent

            if (self.agent_roles[i] is 'leader' and self.agent_types[i] is 'drone'):    # drone-leader

                next_ = ((x + dx), (y + dy))   # Calculate next location

                if self.walls[next_]:
                    next_ = (x, y)              # Do not move into walls

                self.agent_locations[i] = next_
                current_locations[i] = next_    # update current_locations, _collide() disregards collision with drones

                # Set a team's target zone since this is a drone leader
                team_index = self._find_team(i)    # find its team
                if team_index is not None:
                    x,y = self.agent_locations[i]
                    self.target_zones[team_index] = ((x,y),(TARGET_WIDTH,TARGET_HEIGHT))  # set the team's target zone

                    # 5-11-2019 Update leader metrics - num apples in target zone
                    self.apples_targetzone[team_index] = np.sum(np.clip(self.food[x:x+TARGET_WIDTH, y:y+TARGET_HEIGHT],0,1))

            elif self.agent_types[i] is 'crawler':    # crawler-leader or follower

                if self.tagged[i]:   # skip agents that are tagged
                    continue 
            
                next_ = ((x + dx), (y + dy))   # Calculate next location

                if self.walls[next_]:
                    next_ = (x, y)              # Do not move into walls

                # Do not move into the current location of another agent
                if self._collide(i, next_, current_locations):

                    # Debug crawler collision
                    # print ('Collision: agent{}'.format(i))
                    
                    next_ = (x, y)   # If collision, stay in original spot

                self.agent_locations[i] = next_
                current_locations[i] = next_    # update current_locations so that _collide() can work for the other agents

                # Set a team's target zone if the crawler is a team leader 
                if self.agent_roles[i] is 'leader':    # if this is a crawler-leader
                    team_index = self._find_team(i)    # find its team
                    if team_index is not None:
                        x,y = self.agent_locations[i]
                        self.target_zones[team_index] = ((x,y),(TARGET_WIDTH,TARGET_HEIGHT))  # set the team's target zone

                        # 5-11-2019 Update leader metrics - num apples in target zone
                        self.apples_targetzone[team_index] = np.sum(np.clip(self.food[x:x+TARGET_WIDTH, y:y+TARGET_HEIGHT],0,1))
            else:
                raise Exception('Invalid type-role for agent {}: {}-{}'.format(i, self.agent_types[i],self.agent_roles[i]))

        # Debug agent movements
        # print (self.agent_locations)
      
        # (3) Set target and banned zones and generate metrics
        # Now that the agents' positions are final, we place the banned or target zone into the agents' game spaces.
        # Then for the crawler -followers only, we generate the 2 games metrics: in_bannedzone and in_targetzone

        self.BANISH = [np.zeros_like(self.food) for i in range(self.n_agents)]
        self.TARGET = [np.zeros_like(self.food) for i in range(self.n_agents)]

        for i,agent in enumerate(self.agent_locations):  # go through the list of agents

            team_index = self._find_team(i)    # find its team
            x,y = agent

            if team_index is not None: 
                # Set the banned zones if they exist
                if self.banned_zones[team_index] is not None: 
                    (x,y),(width, height) = self.banned_zones[team_index]   # get zone dimen
                    x = x + self.padding + 1  # offset padding
                    y = y + self.padding + 1
                    self.BANISH[i][x:x+width, y:y+height] = 1   # mark banned zone 

                # Set the target zones if they exist
                if self.target_zones[team_index] is not None:
                    (x,y),(width, height) = self.target_zones[team_index]   # get zone dimen
                    self.TARGET[i][x:x+width, y:y+height] = 1   # mark target zone  

            # We generate the metrics in_bannedzone and in_targetzone for crawler-followers only
            if (self.agent_roles[i] is 'follower' and self.agent_types[i] is 'crawler'):    
                # Check if agent is inside the banned zone
                if self.BANISH[i][agent]:
                    self.in_bannedzone[i] = True
                else:
                    self.in_bannedzone[i] = False

                # Check if agent is inside the target zone
                if self.TARGET[i][agent]:
                    self.in_targetzone[i] = True
                else:
                    self.in_targetzone[i] = False

        # Debug target zone
        # print (self.in_targetzone)

 
        # (4) Implement laser tagging and generate metrics
        # This implment laser tagging by crawler agents. Drones cannot be tagged by laser since they are in 
        # the air. 

        self.beams[:] = 0    # Initialize game space for laser beam

        # If action is LASER
        for i, act in enumerate(action_n):

            # Only crawler agents can fire laser
            if self.agent_types[i] is 'crawler':    
                # initialize agent's laser parameters
                self.fire_laser[i] = False
                self.kill_zones[i][:] = 0
                self.US_tagged[i] = 0 
                self.THEM_tagged[i] = 0

                # This updates agent metrics wrt laser firing:
                #  - fire_laser: the agent has fired its laser 
                #  - US_tagged: How many agents of same team has been tagged
                #  - THEM_tagged: How many agents of other teams has been tagged
                # Note that drones cannot be tagged by lasers.
                if act == LASER:
                    self.fire_laser[i] = True       # agent has fired his laser
                    laser_field = self._viewbox_slice(i, CRAWLER_LASER_WIDTH, CRAWLER_LASER_DEPTH, offset=1)
                    self.kill_zones[i][laser_field] = 1  # define the kill zone
                    self.beams[laser_field ] = 1    # place beam on kill zone
                    # register how many US vs THEM agents have been fired upon
                    self.US_tagged[i], self.THEM_tagged[i] = self._laser_hits(self.kill_zones[i], i)
        
        # Debug laser lagging
        # print (self.tagged)
        

        # (5) Implement crawler reward and penalty
        # A bit hard-to-read. If agent lands on a food cell, that cell is set to -15. Then for each 
        # subsequent step, it is incremented by 1 until it reaches 1 again.
        # self.initial_food is the game space created from the text file whereby the cell with food 
        # is given the value of 1 while every other cell has the value of 0.

        # Initialize reward_n, done_n and info_n    
        reward_n = [0 for _ in range(self.n_agents)]
        done_n = [self.done] * self.n_agents
        info_n = [None for _ in range(self.n_agents)]   # initialize agent info
        
        self.food = (self.food + self.initial_food).clip(max=1)   # consumed apples regenerate after 15 steps

        # In case a crawler agent lands on the river, a cell with food, or is tagged
        for i, a in enumerate(self.agent_locations):

            if (self.tagged[i] or 
                self.agent_types[i] is 'drone'):   # Skip if agent is tagged or if agent is a drone
                continue
            
            # Agent lands in the river
            if self.rivers[a] == 1:
                reward_n[i] += self.river_penalty      # Agent is given a penalty for being in the river

                # Debug river
                # print('Agent {} is in river'.format(i))
 
            # Agent lands on a food unit
            if self.food[a] == 1:
                self.food[a] = -15    # Food is respawned every 15 steps once it has been consumed
                reward_n[i] += REWARD_APPLE       # Agent is given reward for gathering an apple
                self.consumption.append((i,a))    # Update consumption history (agent #, location of food)

            # Agent is inside a laser beam
            if self.beams[a]:
                self.tagged[i] = 25   # If agent is tagged, it is removed from the game for 25 steps
                self.agent_locations[i] = (-1,-1)  # and it is sent to Nirvana

        # Respawn agent after 25 steps; tagged should always be between 0 to 25
        for i, tag in enumerate(self.tagged):

            if (self.tagged[i] is None or 
                self.agent_types[i] is 'drone'):   # Skip if agent is a drone
                continue

            # If agent is a crawler
            if tag > 1:   # agent has been tagged
                self.tagged[i] = tag - 1   # count down tagged counter (from 25)
            elif tag == 1:     # When tagged is 1, it is time to respawn agent i
                # But check to make sure no agent at the respawn location
                current_locations = [a for a in self.agent_locations]

                next_ = self.spawn_points[i]
                if self._collide(i, next_, current_locations):
                    self.agent_locations[i] = (-1,-1)    # Stay in Nirvana if there is collision
                else:
                    self.agent_locations[i] = next_      # Otherwise, respawn  
                    self.orientations[i] = UP
                    self.tagged[i] = 0


        # (7) Organize _step() return data
        obs_n = self.state_n    # obs_n is self.state_n

        for i, agent in enumerate(self.agent_locations):
            if (self.agent_types[i] is 'drone'):    # drone
                team_index = self._find_team(i)    # find its team
                if team_index is not None:
                    info_n[i] = self.apples_targetzone[team_index]
                else:
                    raise Exception("Leader agent {} has no team!".format(i))
            elif (self.agent_types[i] is 'crawler'):    # drone
                # 03-02-2019  Add in_bannedzone and in_targetzone as agent metrics
                info_n[i] = (self.tagged[i], self.fire_laser[i], self.US_tagged[i], self.THEM_tagged[i],  \
                    self.in_bannedzone[i], self.in_targetzone[i])

        return obs_n, reward_n, done_n, info_n

    # 5-04-2019 Rewrite for code readibility
    # Generate slice(tuple) to generate observation space or laser kill zone for a crawler agent
    def _viewbox_slice(self, agent_index, width, depth, offset=0):
        
        # This generates the measurements need to generate an observation space for an agent based 
        # on its orientation:
        # e.g. If width is 10, the agent can perceive 5 pixels to the left,  4 pixels to its right.
        # But if width is 5, it perceives 2 pixels to its left and to its right.
        left = width // 2
        right = left if width % 2 == 0 else left + 1
        x, y = self.agent_locations[agent_index]

        # This generates index slices for observation spaces in all 4 orientations
        UP = (slice(x - left, x + right), slice(y - offset, y - offset - depth, -1))
        RIGHT = (slice(x + offset, x + offset + depth), slice(y - left, y + right))
        DOWN = (slice(x + left, x - right, -1), slice(y + offset, y + offset + depth))
        LEFT = (slice(x - offset, x - offset - depth, -1), slice(y + left, y - right, -1))

        views = [UP,RIGHT,DOWN,LEFT]         

        # return only the index slices for the specific agent's actual orientation
        return views[self.orientations[agent_index]]  


    # state_n (next state) is a property object. So this function is run everytime state_n is
    # called as a variable.
    @property
    def state_n(self):


        food = self.food.clip(min=0)   # Mark the food's location; self.food has dim 100x60

        # Create game spaces for agent locating US vs THEM crawler agents 
        US = [np.zeros_like(self.food) for i in range(self.n_agents)]
        THEM = [np.zeros_like(self.food) for i in range(self.n_agents)]

        # 5-05-2019 state_n will be a list of agents' states

        # Zero out next states for the agents
        # s = np.zeros((self.n_agents, self.viewbox_width, self.viewbox_depth, NUM_FRAMES))

        states = []
        
        for i, (x, y) in enumerate(self.agent_locations):   # Go through each agent

            # Construct the full state for the game - 7 frames denoting:
            # 1. Location of Food
            # 2. Location of US agents in the viewbox
            # 3. Location of THEM agents in the viewbox
            # 4. Location of the walls
            # 5. Location of the river(s)
            # 6. Location of banned zone   (added to aid exploration and lead-follow)
            # 7. Location of target zone   (added to aid exploration and lead-follow)

            if self.agent_types[i] is 'drone':

                # go through the list of crawler agents and mark them as US or THEM
                for j, loc in enumerate(self.agent_locations):
                    if (self.tagged[j] is 0                      # if the crawler agent is in the game (not tagged out)
                        and self.agent_types[j] is not 'drone'):     
                        # compare the two crawler agents' teams
                        if self.agent_teams[i] == self.agent_teams[j]: 
                           US[i][loc] = 1     # Mark US agent's location
                            # For debug only
                            # print ('Agent{} of team {} is US of team {}'.format(j, self.agent_teams[j], self.agent_teams[i]))
                        else:
                            THEM[i][loc] = 1     # Mark THEM agent's location
                            # For debug only
                            # print ('Agent{} team {} is THEM of team{}'.format(j, self.agent_teams[j], self.agent_teams[i]))

                full_state = np.stack([food, US[i], THEM[i], self.walls, self.rivers, self.BANISH[i], self.TARGET[i]], axis=-1)

                # Drone agent can observe the entire game space
                observation = np.asarray(full_state)
                states.append(observation)

            elif self.agent_types[i] is 'crawler':

                # go through the list of crawler agents and mark them as US or THEM
                for j, loc in enumerate(self.agent_locations):
                    if (self.tagged[j] is 0                      # if the crawler agent is in the game (not tagged out)
                        and self.agent_types[j] is not 'drone'):     
                        # compare the two crawler agents' teams
                        if self.agent_teams[i] == self.agent_teams[j]:
                            US[i][loc] = 1     # Mark US agent's location
                            # For debug only
                            # print ('Agent{} of team {} is US of team {}'.format(j, self.agent_teams[j], self.agent_teams[i]))
                        else:
                            THEM[i][loc] = 1     # Mark THEM agent's location
                            # For debug only
                            # print ('Agent{} team {} is THEM of team{}'.format(j, self.agent_teams[j], self.agent_teams[i]))


                full_state = np.stack([food, US[i], THEM[i], self.walls, self.rivers, self.BANISH[i], self.TARGET[i]], axis=-1)

                if self.tagged[i] is 0:  # if crawler agent has not been tagged

                    # Create partial observation space for Crawler agent using _viewbox_slice()
                    xs, ys = self._viewbox_slice(i, self.viewbox_width, self.viewbox_depth)
                    observation = np.asarray(full_state[xs, ys, :])
                    # Orient the observation space correctly
                    states.append(observation if self.orientations[i] in [UP, DOWN] else observation.transpose(1, 0, 2))
                else:
                    states.append(np.zeros((self.viewbox_width, self.viewbox_depth, NUM_FRAMES)))

            else:
                raise Exception('Unknown agent type: {}'.format(agent_types[i]))

            # Debug
            # print (observation.shape)            

        return states # Return the agents' observations


    # To reset the environment
    def _reset(self):

        # Build food stash
        self.food = self.initial_food.copy()

        # Build wall around the game space defined by the map file.
        p = self.padding
        self.walls[p:-p, p] = 1
        self.walls[p:-p, -p - 1] = 1
        self.walls[p, p:-p] = 1
        self.walls[-p - 1, p:-p] = 1

        self.beams = np.zeros_like(self.food)  # game space for the laser beams

        # 5-07-2019
        # A crawler agent has orientation. A drone agent does not.
        # Orientation is set to UP for crawlers, None for drones
        self.orientations =  [UP if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]  

        # At reset, the agents are relocated to their spawn points
        for i, spawn in enumerate(self.spawn_points):
            spawn_x, spawn_y = spawn
            self.agent_locations[i] = (spawn_x, spawn_y)

        # 4-29-2019
        # Go through the list of agents. If the agent is a leader, set it location as target zone for its team
        for i, agent in enumerate(self.agent_locations):
            if self.agent_roles[i] is 'leader':    # if agent is a leader
                team_index = self._find_team(i)    # find its team
                if team_index is not None:
                    x,y = agent
                    self.target_zones[team_index] = ((x,y),(TARGET_WIDTH,TARGET_HEIGHT))  # set the team's target zone

        # 03-01-2019 
        # Create game spaces for target zone and banned zone (for each agent)
        self.BANISH = [np.zeros_like(self.food) for i in range(self.n_agents)]
        self.TARGET = [np.zeros_like(self.food) for i in range(self.n_agents)]
        
        for i, agent in enumerate(self.agent_locations):  # go through the list of agents
            team_index = self._find_team(i)    # find its team
            if team_index is not None:

                # Set the banned zones if they exist
                if self.banned_zones[team_index] is not None: 
                    (x,y),(width, height) = self.banned_zones[team_index]   # get zone dimen
                    x = x + self.padding + 1  # offset padding
                    y = y + self.padding + 1
                    self.BANISH[i][x:x+width, y:y+height] = 1   # mark banned zone 
        
                # Set the target zones if they exist
                if self.target_zones[team_index] is not None:
                    (x,y),(width, height) = self.target_zones[team_index]   # get zone dimen
                    self.TARGET[i][x:x+width, y:y+height] = 1   # mark target zone        

        # 5-07-2019
        # A crawler agent has laser and zone parameters. A drone agent does not.

        # Crawler agent's Laser parameters
        self.tagged =  [0 if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]    # Tagged = False
        self.fire_laser =  [False if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]    # Fire Laser = False
        self.kill_zones =  [np.zeros_like(self.food) if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]    # laser kill zones
        self.US_tagged =  [0 if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]    # agents of same team tagged = 0
        self.THEM_tagged =  [0 if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]    # agents of different teams tagged = 0

        # Crawler agent's zone parameters
        self.in_bannedzone =  [0 if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]    # Inside Banned Zone = False
        self.in_targetzone =  [0 if agent_type is 'crawler' else None 
                                    for agent_type in self.agent_types]    # Inside Target Zone = False

        # 5-11-2019 Leader (crawler or drone) metrics
        self.apples_targetzone = [0 for team in self.teams]   # Num of apples in team's targetzone
        self.US_targetzone = [0 for team in self.teams]   # Num of US crawlers in team's targetzone

        self.consumption = []    # a list for keep track of consumption history

        return self.state_n  # Since state_n is a property object, so it will call function _state_n()


    # To close the rendering window
    def _close_view(self):
        # If rendering window is active, close it
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
        self.done = True   # The episode is done
    

    # TO render the game    
    def _render(self, mode='human', close=False):
        if close:
            self._close_view()
            return

        # The canvas is defined by the game space
        canvas_width = self.gamespace_width * self.scale
        canvas_height = self.gamespace_height * self.scale

        if self.root is None:
            self.root = tk.Tk()
            self.root.title('Gathering')
            self.root.protocol('WM_DELETE_WINDOW', self._close_view)
            self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
            self.canvas.pack()

        self.canvas.delete(tk.ALL)
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill='black')


        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * self.scale,
                y * self.scale,
                (x + 1) * self.scale,
                (y + 1) * self.scale,
                fill=color,
            )

        def draw_zone(x, y, width, height, color):
            self.canvas.create_rectangle(
                x * self.scale,
                y * self.scale,
                (x + width) * self.scale,
                (y + height) * self.scale,
                outline=color
            )
     
        # Refresh the canvas by placing pixels for laser beams, food units and walls        
        for x in range(self.gamespace_width):
            for y in range(self.gamespace_height):
                if self.beams[x, y] == 1:
                    fill_cell(x, y, 'yellow')
                if self.food[x, y] == 1:
                    fill_cell(x, y, 'green')
                if self.walls[x, y] == 1:
                    fill_cell(x, y, 'grey')
                if self.rivers[x, y] == 1:
                    fill_cell(x, y, 'Aqua')

        # 5-05-2019
        # Place Crawler agents on the canvas, but do not render Drone agents
        for i, (x, y) in enumerate(self.agent_locations):

            if (self.tagged[i] is 0                      
                and self.agent_types[i] is not 'drone'):     # provided agent i has not been tagged and is not a drone 
                fill_cell(x, y, self.agent_colors[i])            

        # Refresh the canvas by placing the target and banned zone by team
        if self.target_zones is not None:
            for zone in self.target_zones:
                if zone is not None:
                    (x,y),(width, height) = zone
                    draw_zone(x, y, width, height, 'green')    # 04-29-2019 Remove padding offset for target zone

        if self.banned_zones is not None:
            for zone in self.banned_zones:
                if zone is not None:
                    (x,y),(width, height) = zone
                    draw_zone(x + self.padding + 1, y + self.padding + 1, width, height, 'red')  

        if True:
            # Update Total Rewards
            self.canvas.create_text(canvas_width/2,20,fill="darkblue",font="Times 15 italic bold",
                        text="The Score:")

        if self.debug_window:
            # 5-31-2019 Implement self.debug_window flag
            # Debug view: see the debug agent's viewbox perspective.
            p1_state = self.state_n[self.debug_agent].reshape(self.viewbox_width, self.viewbox_depth, NUM_FRAMES)
            for x in range(self.viewbox_width):
                for y in range(self.viewbox_depth):
                    food, us, them, wall, river, banned, target = p1_state[x, y]
                    # 03-04-2019 commented out because of assertion
                    # assert sum((food, us, them, wall, river)) <= 1
                    y_ = self.viewbox_depth - y - 1
                    if food:
                        fill_cell(x, y_, 'green')
                    elif us:
                        fill_cell(x, y_, 'cyan')
                    elif them:
                        fill_cell(x, y_, 'red')
                    elif wall:
                        fill_cell(x, y_, 'gray')
                    elif river:
                        fill_cell(x, y_, 'aqua')
                    elif target:
                        fill_cell(x, y_, 'gray26')
                    elif banned:
                        fill_cell(x, y_, 'maroon4')
            self.canvas.create_rectangle(
                0,
                0,
                (self.viewbox_width + 1)* self.scale,
                (self.viewbox_depth + 1) * self.scale,
                outline='blue',
            )

        self.root.update()


    # To close the environment
    def _close(self):
        self._close_view()

    # To delete the environment
    def __del__(self):
        self.close()


_spec = {
    'id': 'xTeam-Luke-v64',
    'entry_point': CrossingEnv,
    'reward_threshold': 500,   # The environment threshold at 100 appears to be too low
}

gym.envs.registration.register(**_spec)
