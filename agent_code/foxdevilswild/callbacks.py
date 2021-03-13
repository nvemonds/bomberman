import os
import pickle
import random
from itertools import product

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    # todo: define list of states based on game_state

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)

def getState(self,game_state):
    return int_states

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
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

    self.logger.debug("Querying model for action.")

    #return action of current state with greatest Q value
    # Forward
    # return np.random.choice(ACTIONS, p = self.model)

    row           = self.Q[status,:]
    action_index  = np.argmax(row)
    action        = ACTIONS[action]
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

    """ Example Code
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # For example, you could construct several channels of equal shape, ...
    channels = []
    channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    stacked_channels = np.stack(channels)
    # and return them as a vector

    return stacked_channels.reshape(-1)
    """

    # List of coin positions
    coin_positions      = game_state['coins']
    own_position       = game_state['self'][-1]

    # List of relative coin positions
    relative_coin_positions = [(item[0]- own_position[0],item[1] - own_position[1]) for item in coin_positions]

    # Calculate Radii of the coins
    fun   = lambda x : np.sqrt(x[0]**2 + x[1]**2)
    radii = map(fun, relative_coin_positions)

    # Definiere nächstes target als das mit der kleinsten Distanz
    next_destination = relative_coin_positions[np.argmax(radii)]

    # Diskretisiere die Distanz zum nächsten Element (noch zu verbessern)
    list_of_possible_discretized_radii = list(range(0,22))
    discretized_radius                 = int(radii)

    ## Calculiere den Winkel zu dem Element
    # Calculiere
    winkel           = np.arctan2(y,x)
    # Diskretisiere den Winkel der zwischen -pi/2 und pi/2
    discrete_intervalls = np.linspace(-np.pi/2, np.pi/2, num=17)
    assert len(discrete_intervalls) == 17, "Die Diskretisierung der Winkel hat nicht funktioniert."
    # Gibt eine Liste der Mean Winkel in jedem Intervall an
    mean_winkels         = [(discrete_intervalls[index + 1]-discrete_intervalls[index])/2 for index in range(len(discrete_intervalls)-1)]


    try:
        del winkel
    except:
        None
    for index in range(len(discrete_intervalls)):
        if discrete_intervalls[index] < winkel  and winkel < discrete_intervalls[index+1]:
            break
    discretized_winkel = index

    # Hier wird ein Zustandsdict definiert, welches den Zustandsindex auf die Kombination (Radius, Winkel)
    # abbildet. Das dient vor allem der Nachverfolung.
    zustandsdict = {}
    reverse_dict = {}
    counter      = 0
    
    for mean_winkel, radius in product(mean_winkels,list_of_possible_discretized_radii):
        zustandsdict[counter]               = (mean_winkel, radius)
        reverse_dict[(mean_winkel, radius)] = counter
        counter                            += 1

    assert len(zustandsdict.keys()) == 16*2, "Die berechneten Zustände stimmen nicht mit der initierten Matrix 16*2 zusammen"

    # Hier ist das finale Resultat:
    state = reverse_dict[(discretized_winkel,discretized_radius)]
    return state
