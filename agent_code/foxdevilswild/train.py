import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    # todo: initialize  Q-matrix of size n*6 to 0 where n is number of states
    # todo: initialize list of rewards
    self.rewards = {
    'MOVED_LEFT': 0,
    'MOVED_RIGHT': 0,
    'MOVED_UP' : 0,
    'MOVED_DOWN' : 0,
    'WAITED' : 0,
    'INVALID_ACTION': -50,

    'BOMB_DROPPED' : 0,
    'BOMB_EXPLODED' : 0,

    'CRATE_DESTROYED' :0,
    'COIN_FOUND' : 0,
    'COIN_COLLECTED' : 40,

    'KILLED_OPPONENT' : 0,
    'KILLED_SELF' : 0,

    'GOT_KILLED': 0,
    'OPPONENT_ELIMINATED' : 0,
    'SURVIVED_ROUND' : 0
    }
    # todo: initialize learning rate and discount rate
    self.learning_rate = 0.3
    self.discount_rate = 0.7

    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    with open("my-saved-model.pt", "rb") as file:
        self.Q = pickle.load(file)



def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    if self_action == None:
        return None

    reward = 0
    """
    coin_dist_reduced = len(new_game_state["coins"])>0 and \
                        np.min(np.linalg.norm(old_game_state["coins"] - np.array(old_game_state['self'][-1]), axis=1)) !=1 and\
                         (np.min(np.linalg.norm(old_game_state["coins"] - np.array(old_game_state['self'][-1]), axis = 1)) >\
                          np.min(np.linalg.norm(new_game_state["coins"] - np.array(new_game_state['self'][-1]), axis = 1)))"""

    if len(new_game_state["coins"])>0:
        reward += np.sign((np.min(np.linalg.norm(old_game_state["coins"] - np.array(old_game_state['self'][-1]), axis = 1)) -\
                          np.min(np.linalg.norm(new_game_state["coins"] - np.array(new_game_state['self'][-1]), axis = 1))))*5*(np.min(np.linalg.norm(old_game_state["coins"] - np.array(old_game_state['self'][-1]), axis = 1)) /\
                          np.min(np.linalg.norm(new_game_state["coins"] - np.array(new_game_state['self'][-1]), axis = 1)))


    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    # todo: set old game state based on old_game_state
    old_state = state_to_features(old_game_state)
    # todo: set current game state based on new_game_state
    new_state = state_to_features(new_game_state)
    # todo: update Q-values : Q[old state, action] = Q[old state, action] + lr * (reward + gamma * np.max(Q[new state, :]) — Q[old state, action])
    reward += np.sum([self.rewards[i] for i in events])
    action = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'].index(self_action)
    #print("reward: ",reward)
    #print("events:", events)
    self.Q[old_state, action] += self.learning_rate * (reward + self.discount_rate*np.max(self.Q[new_state, :])-self.Q[old_state, action])
    #print(self.Q[old_state, action])
    # Idea: Add your own events to hand out rewards
    #if ...:
    #    events.append(PLACEHOLDER_EVENT)

    # state_to_features is defined in callbacks.py
    #self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    if last_action == None:
        return None
    old_state = state_to_features(last_game_state)
    reward = np.sum([self.rewards[i] for i in events])
    action = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB'].index(last_action)
    #print("events:", events)
    self.Q[old_state, action] += self.learning_rate * (reward + self.discount_rate*np.max(self.Q[old_state, :])-self.Q[old_state, action])
    """
    if self.learning_rate * (reward + self.discount_rate*np.max(self.Q[old_state, :])-self.Q[old_state, action]) <0:
        print(events)"""
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.Q, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
