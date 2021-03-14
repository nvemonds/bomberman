import os
import pickle
import random

import numpy as np
from itertools import  product
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
norm = lambda x: np.sqrt(x[0]**2+x[1]**2)
discretized_angles =np.arange(17)

# zustandsdict
zustandsdict = {}
counter = 0

for angle, borderstatus, bombstatus in product(discretized_angles,[0,1,2],[0,1]):
    zustandsdict[counter] = (angle, borderstatus, bombstatus)
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

    if (self.train or not os.path.isfile("my-saved-model.pt")) and 0==1:
        self.logger.info("Setting up model from scratch.")
        self.Q = np.random.rand(17 * 3*2, 6)*100
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
    random_prob = 0.1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    #load Q matrix from last training process
    #self.Q = np.load("my-saved-model.pt", allow_pickle=True)

    action_index  = np.argmax(self.Q[status,:])
    action        = ACTIONS[action_index]
    #print(self.Q[status,:])
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
    #the following block of code refers to the state property of coin finding
    coin_positions      = game_state['coins']
    if any(coin_positions):
        relative_coin_positions = coin_positions-own_position
        radii =  np.apply_along_axis(norm,1,relative_coin_positions)
        minimal_radius = np.argmin(radii)
        # Calculate angle w.r.t nearest coin
        x = relative_coin_positions[minimal_radius][0]
        y = relative_coin_positions[minimal_radius][1]
        angle = np.arctan2(x, y)+np.pi
        discretized_angle = int(angle *2.545)
        #print(discretized_angle)
    else:
        discretized_angle = 16

    #the following block of code refers to the state property of invalid movements
    borderstatus = 0
    if not game_state["field"][tuple(own_position-[0,1])] == 0:# or game_state["field"][tuple(own_position-[0,1])] == 1:
        borderstatus =1
    if not game_state["field"][tuple(own_position-[1,0])] == 0:# or game_state["field"][tuple(own_position-[1,0])] == 1:
        borderstatus =2

    #the following block of code refers to the state property of placing bombs
    bombstatus = 0
    bombs = np.array(sum(np.array(game_state["bombs"]).reshape((len(game_state["bombs"]), 2))[:, 0], ())).reshape(len(game_state["bombs"]),2)

    if bombs.any() and (own_position-bombs).flatten().any() and np.apply_along_axis(norm,1,own_position - bombs).any()<4:
        bombstatus = 1
    #print("state: ", reverse_dict[(discretized_angle, borderstatus, bombstatus)])
    return reverse_dict[(discretized_angle, borderstatus, bombstatus)]
