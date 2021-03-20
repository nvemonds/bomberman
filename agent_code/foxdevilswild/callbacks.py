import os
import pickle
import random

import numpy as np
from itertools import  product
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
norm = lambda x: np.sqrt(x[0]**2+x[1]**2)

# zustandsdict
zustandsdict = {}
counter = 0

for c,d, e, f, g, h in product(range(9),range(3), range(3), range(3), range(3), range(4)):
    zustandsdict[counter] = (c,d, e, f, g, h)
    counter += 1

reverse_dict = {v: k for k, v in zustandsdict.items()}


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.Q = np.zeros((9*3*3*3*3*4, 6))
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(self.Q, file)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.Q = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    # todo: set current state based on game_state
    status = state_to_features(game_state)

    # todo Exploration vs exploitation
    random_prob = 0.2
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .0, .2])

    self.logger.debug("Querying model for action.")
    action_index  = np.argmax(self.Q[status, :])
    action        = ACTIONS[action_index]

    #print(self.Q[status, :])
    return action


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    own_position = np.array(game_state['self'][-1])
    field_status = game_state["field"]
    #the following block of code refers to the state property of coin finding
    coin_positions = game_state['coins']

    if any(coin_positions):
        relative_coin_positions = coin_positions-own_position
        radii =  np.apply_along_axis(norm,1,relative_coin_positions)
        if np.min(radii) != 0:
            minimal_radius = np.argmin(radii)
            # Calculate angle w.r.t nearest coin
            x = relative_coin_positions[minimal_radius][0]
            y = relative_coin_positions[minimal_radius][1]
            angle = np.arctan2(x, y)+np.pi
            coin_angle = np.round(angle/(np.pi/4))
        else:
            coin_angle = 0
    else:
        coin_angle = 0

    #the following block of code refers to the state property of neighboring tiles
    up, down, left, right = 0, 0, 0, 0
    crate_angle = 0
    crate_distance = 0
    if (field_status == 1).any():
        relative_crate_positions = np.argwhere(field_status==1)-own_position
        nearest_crate = np.argmin(np.apply_along_axis(norm,1,relative_crate_positions))
        x = relative_crate_positions[nearest_crate][0]
        y = relative_crate_positions[nearest_crate][1]
        crate_angle = np.round(np.arctan2(x, y)/(np.pi/2)+2)
        crate_distance = int(norm([x,y]))

        """if field_status[tuple(own_position - [0,1])] == 1:
            crate_distance += 1
        if field_status[tuple(own_position - [1,0])] == 1:
            crate_distance += 1
        if field_status[tuple(own_position + [0,1])] == 1:
            crate_distance += 1
        if field_status[tuple(own_position + [1,0])] == 1:
            crate_distance += 1"""


    #the following block of code refers to the state property of invalid movements and explosions
    bombs = np.array(sum(np.array(game_state["bombs"],dtype=object).reshape((len(game_state["bombs"]), 2))[:, 0], ())).reshape(
        len(game_state["bombs"]), 2)
    others = np.array(sum(np.array(game_state["others"],dtype=object).reshape((len(game_state["others"]), 4))[:, 3], ())).reshape(
        len(game_state["others"]), 2)
    if field_status[tuple(own_position - [0,1])] == -1 or game_state["explosion_map"][tuple(own_position - [0,1])]!= 0 \
            or not np.linalg.norm(np.array(own_position) - [0,1] - bombs, axis = 1).all() or not np.linalg.norm(np.array(own_position) - [0,1] - others, axis = 1).all():
        up = 1
    elif field_status[tuple(own_position - [0,1])] == 1:
        up = 2
    if field_status[tuple(own_position - [1,0])] == -1 or game_state["explosion_map"][tuple(own_position - [1,0])] != 0\
            or not np.linalg.norm(np.array(own_position) - [1,0] - bombs, axis = 1).all() or not np.linalg.norm(np.array(own_position) - [1,0] - others, axis = 1).all():
        left = 1
    elif field_status[tuple(own_position - [1, 0])] == 1:
        left = 2
    if field_status[tuple(own_position + [0,1])] == -1 or game_state["explosion_map"][tuple(own_position + [0,1])] != 0\
            or not np.linalg.norm(np.array(own_position) + [0,1] - bombs, axis = 1).all() or not np.linalg.norm(np.array(own_position) + [0,1] - others, axis = 1).all():
        down = 1
    elif field_status[tuple(own_position + [0, 1])] == 1:
        down = 2
    if field_status[tuple(own_position + [1,0])] == -1 or game_state["explosion_map"][tuple(own_position + [1,0])] != 0\
            or not np.linalg.norm(np.array(own_position) + [1,0] - bombs, axis = 1).all() or not np.linalg.norm(np.array(own_position) + [1,0] - others, axis = 1).all():
        right = 1
    elif field_status[tuple(own_position + [1, 0])] == 1:
        right = 2

    #the following block of code refers to the state property of placing bombs
    bombstatus = 0
    if bombs.any() and np.max(np.apply_along_axis(norm,1,bombs-own_position))<5:
        bombstatus = 1
        if own_position[0] % 2 and ((bombs-own_position)[:,0]==0).any():
            bombstatus = 2
        elif own_position[1] % 2 and ((bombs-own_position)[:,1]==0).any():
            bombstatus = 3
    #print(coin_angle, up, down, left, right)
    #print(coin_angle)
    return reverse_dict[(coin_angle, up, down, left, right, bombstatus)]
