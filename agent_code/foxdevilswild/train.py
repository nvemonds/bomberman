import os
import pickle
import random
import numpy as np
from typing import List

import events as e
from .callbacks import state_to_features, regression_model, explosion_radius

# Hyper parameters
FEATURE_DIMS = 26 #not a hyperparameter but dont wanna hard code it
GAMMA = 0 #discount factor
LEARNING_RATE = 0#1e-2
NUMBER_OF_TRAINING_INSTANCES = 3000
EPS_MIN = 0#0.05
EPS_MAX = 0#1
EPS_DEC = 0#EPS_MIN**(1/(0.9*NUMBER_OF_TRAINING_INSTANCES))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_DICT = {None:6,'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}

def setup_training(self):
	"""
	Initialise self for training purpose.

	This is called after `setup` in callbacks.py.

	:param self: This object is passed to all callbacks and you can set arbitrary values.
	"""
	#Setup an array that will note transition tuples
	
	#(s, a, s', r)
	self.transitions = []
	
	#make an array for the stats (for plots) and initialize exploration rate
	if os.path.isfile('stats.npy'):
		self.stats = np.load('stats.npy')
		self.instance = self.stats[-1,0] + 1
		self.eps = np.max([EPS_MIN, EPS_DEC**self.instance])
	else:
		self.stats = np.array([0, 0, 0, 0, 0, 0]) #instance, rounds survived, coins collected, crates destroyed, opponents killed, steps taken
		np.save('stats.npy', self.stats)
		self.instance = 0
		self.eps = EPS_MAX
	self.rounds_survived = 0
	self.coins_collected = 0
	self.crates_destroyed = 0
	self.opponents_killed = 0
	self.steps_taken = 0
	
	return

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
	
	if "COIN_COLLECTED" in events: self.coins_collected += 1
	if "OPPONENT_ELIMINATED" in events: self.opponents_killed += events.count("OPPONENT_ELIMINATED")
	if "CRATE_DESTROYED" in events: self.crates_destroyed += events.count("CRATE_DESTROYED")
	
	if old_game_state != None and new_game_state != None:
		self_old = old_game_state['self'][3]
		self_new = new_game_state['self'][3]
		coins = old_game_state['coins']
		board = old_game_state['field']
		walls = board.copy()
		walls[walls != -1] = 0
		bombs = old_game_state['bombs']
		others = old_game_state['others']
		others_pos = np.empty((len(others),2), dtype=int)
		for i in range(len(others)): others_pos[i] = others[i][3]
		non_passable_tiles = board.copy()
		if len(others) > 0: non_passable_tiles[(others_pos[:,0], others_pos[:,1])] = 2
		
		#approached coin?
		if len(coins) > 0:
			found, dist_old, nearest = dist_to_nearest(tuple(self_old), non_passable_tiles, coins)
			_, dist_new, _ = dist_to_nearest(tuple(self_new), non_passable_tiles, np.array(nearest))
			if dist_old > dist_new and found: 
				events.append("APPROACHED_COIN")
		
		#approached crate?
		crate_walls = non_passable_tiles.copy()
		crate_walls[crate_walls == 1] = 0
		crate_pos = np.array(np.where(board == 1)).T
		if len(crate_pos) > 0:
			found, dist_old, nearest = dist_to_nearest(tuple(self_old), crate_walls, crate_pos)
			_, dist_new, _ = dist_to_nearest(tuple(self_new), crate_walls, np.array(nearest))
			if dist_old > dist_new and found: 
				events.append("APPROACHED_CRATE")

		#escaping death?
		if len(bombs) > 0: 
			nogo = board.copy()
			oob = lambda x: not ( (np.array(x) >= 0).all() and (np.array(x) < 17).all() ) #out of bounds
			for b in bombs:
				nogo[tuple(b[0])] = 7
				explosions = explosion_radius(b[0], board)
				nogo[(explosions[:,0],explosions[:,1])] = 7
			#calculate coordinates of free tiles
			free_pos = np.array(np.where(nogo == 0)).T
			#calculate which way to go to
			_, dist_old, nearest = dist_to_nearest(tuple(self_old), non_passable_tiles, free_pos)
			_, dist_new, _ = dist_to_nearest(tuple(self_new), non_passable_tiles, np.array(nearest))
			if nogo[tuple(self_old)] == 7 and dist_old > dist_new:
				events.append("ESCAPING_DEATH")
			#moved into death zone?
			if nogo[tuple(self_old)] == 0 and nogo[tuple(self_new)] == 7:
				events.append("MOVED_TO_DEATH")
			
		#approaching_opponent?
		if len(others) > 0:
			#check
			found, dist_old, nearest = dist_to_nearest(tuple(self_old), board, others_pos)
			_, dist_new, _ = dist_to_nearest(tuple(self_new), board, np.array(nearest))
			if dist_old > dist_new and found:
				events.append("APPROACHED_OPPONENT")
	
	#placed bomb in a good spot (at least one target in range)?
	if "BOMB_DROPPED" in events:
		#extract positions
		others_pos = np.empty((len(others),2), dtype=int)
		for i in range(len(others)): others_pos[i] = others[i][3]
		explosions = explosion_radius(self_new, board)
		if len(others) > 0:
			board[(others_pos[:,0], others_pos[:,1])] = 1
		for e in explosions:
			if board[tuple(e)] == 1:
				events.append("WELL_BOMBED")
				break
	
	#we have to use self.last_action for the action otherwise invalid actions wont get penalized
	if old_game_state != None and new_game_state != None:
		reward = reward_from_events(self, events)
		feats = state_to_features(old_game_state)
		self.transitions.append([feats, self.last_action, state_to_features(new_game_state), reward])
		#rotations by 1,2,3*90° are also valid game states
		self.logger.debug(f'Original feats: {feats}')
		for i in range(1,4):
			r_feats = rotate_features(state_to_features(old_game_state), i)
			r_act = rotate_action(self.last_action, i)
			self.logger.debug(f'Rotated feats: {r_feats}')
			self.logger.debug(f'Rotated action: {r_act}')
			self.transitions.append([r_feats, r_act, rotate_features(state_to_features(new_game_state), i), reward])
	return

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
	"""
	Called at the end of each game or when the agent died to hand out final rewards.

	This is similar to reward_update. self.events will contain all events that
	occurred during your agent's final step.

	This is *one* of the places where you could update your agent.
	This is also a good place to store an agent that you updated.

	:param self: The same object that is passed to all of your callbacks.
	"""
	#self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
	
	#separate the necessary arrays
	T_tau = len(self.transitions)
	prev_states = np.array([])
	actions = np.array([])
	next_states = np.array([])
	rewards = np.array([])
	for i in range(T_tau):
		ith_timestep = self.transitions[i]
		prev_states = np.append(prev_states, ith_timestep[0])
		actions = np.append(actions, ith_timestep[1])
		next_states = np.append(next_states, ith_timestep[2])
		rewards = np.append(rewards, ith_timestep[3])
	prev_states = prev_states.reshape((T_tau,int(len(prev_states)/T_tau)))
	next_states = next_states.reshape((T_tau,int(len(next_states)/T_tau)))
	
	#use those to calculate the values for each timestep (we use td q learning here)
	values = td_learning(self, rewards, next_states)
	#values = mc(rewards)
	#values = q_learning(self, prev_states, next_states, rewards, actions)
	
	self.logger.debug(f'END OF ROUND: Updating model with estimated values {values} of {len(values)} steps')
	
	#use to update model
	update_model(self, prev_states, actions, values)

	# Store the model
	with open("my-saved-model.pt", "wb") as file:
		pickle.dump(self.model, file)

	#update stats
	self.instance+=1
	self.rounds_survived = last_game_state['step']
	if "SURVIVED_ROUND" in events: self.rounds_survived = 400
	if "COIN_COLLECTED" in events: self.coins_collected += 1
	if "OPPONENT_ELIMINATED" in events: self.opponents_killed += events.count("OPPONENT_ELIMINATED")
	if "CRATE_DESTROYED" in events: self.crates_destroyed += events.count("CRATE_DESTROYED")
	self.steps_taken = last_game_state['step']
	self.stats = np.vstack((self.stats, [self.instance, self.rounds_survived, self.coins_collected, self.crates_destroyed, self.opponents_killed, self.steps_taken]))
	np.save('stats.npy', self.stats)
	self.coins_collected = 0
	self.crates_destroyed = 0
	self.opponents_killed = 0
	#decrease epsilon
	self.eps = np.max([self.eps*EPS_DEC, EPS_MIN])

	#clear transitions for next episode
	del self.transitions
	self.transitions = []
	return

def reward_from_events(self, events: List[str]) -> float:
	"""
	*This is not a required function, but an idea to structure your code.*

	Here you can modify the rewards your agent get so as to en/discourage
	certain behavior.
	"""
	game_rewards = {
		"MOVED_LEFT"			: -2,
		"MOVED_RIGHT"			: -2,
		"MOVED_UP"				: -2,
		"MOVED_DOWN"			: -2,

		"WAITED"				: -1,
		"INVALID_ACTION"		: -5,

		"APPROACHED_CRATE"		: 10,
		"BOMB_DROPPED"			: -5,
		"WELL_BOMBED"			: 10,
		
		"BOMB_EXPLODED"			: 0,
		"CRATE_DESTROYED"		: 0,
		"COIN_FOUND"			: 0,

		"APPROACHED_COIN"		: 17,
		"COIN_COLLECTED"		: 0,

		"KILLED_OPPONENT"		: 0,
		"APPROACHED_OPPONENT"	: 7,

		"MOVED_TO_DEATH"		: -90,
		"KILLED_SELF"			: 0,
		"GOT_KILLED"			: 0,

		"OPPONENT_ELIMINATED"	: 0,

		"ESCAPING_DEATH"		: 22,
		"SURVIVED_ROUND" 		: 0
	}
	reward_sum = 0
	for event in events:
		if event in game_rewards:
			reward_sum += game_rewards[event]
	self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
	return reward_sum

def td_learning(self, rewards, next_states_features):
	T = len(rewards)
	y = np.zeros(T)
	next_states_features.reshape((-1,FEATURE_DIMS))
	for t in range(T):
		y[t] = rewards[t] + GAMMA*regression_model(self, next_states_features[t]).max()
	return y

def update_model(self, features, actions, response):
	for a in ACTIONS:
		#separate training data by action
		mask = actions == a
		#get responses for training data with old model
		Y = regression_model(self, features[mask,:], action=a)
		#update the corresponding vector
		N_a = np.sum(mask)
		if N_a > 0:
			self.model[ACTIONS_DICT[a]] += LEARNING_RATE/N_a*np.sum(features[mask].T*(response[mask] - Y), axis = 1)
	return

def dist_to_nearest(start: tuple, obstructions: np.array, targets: np.array):
	'''
	finds the best way to go by applying dyjkstras algorithm
	:start: tuple of starting coordinates
	:obstructions: np.array of the whole board with 0 where there is a free tile
	:targets: np.array of coordinates of target positions
	:ans: np.array of features (1 if this way is the ideal way, 0 if not)
	'''
	targets = np.array(targets).reshape((-1,2))
	#board dimensions
	dim = obstructions.shape
	#some ausiliary functions
	def isin(pos, array):
		ans = False
		for e in array:
			if e[0] == pos[0] and e[1] == pos[1]: 
				ans = True
				break
		return ans

	def within_bounds(pos):
		return (np.array(pos) < np.array(dim)).all() and (np.array(pos) >= 0).all()
	
	#some auxiliary stacks and dicts
	inspecting = [start]
	done = []
	parents = {start:start}
	distances = {start: 0}
	translations = [(0,-1), (1,0), (0,1), (-1,0)] #up, right, down, left
	found_target = False
	
	#if there arent any coins return bullshit (but so that it doesnt break anything :) )
	if len(targets) == 0: return False, 0, np.array([0, 0])
	
	#if not find the shortest way to the closest coin
	while len(inspecting) > 0:
		#check all the tiles in consideration
		relevant_dists = []
		for i in range(len(inspecting)):
			relevant_dists.append(distances[inspecting[i]])
		#now consider the tile with the smallest distance from the start
		current = inspecting.pop(np.argmin(relevant_dists))
		#append it to current so it wont get considered again
		done.append(current)
		#break if that is a coin
		if isin(current, targets): 
			found_target = True
			break
		#now give all the neighbors a distance based on the distance of the current tile
		for t in translations:
			neigh = (current[0]+t[0], current[1]+t[1])
			if within_bounds(neigh) and obstructions[neigh] == 0: #check if that is a free tile
				if neigh not in inspecting and neigh not in done: inspecting.append(neigh) #check if this tile hasnt been used yet
				if neigh not in distances or distances[neigh] > distances[current] + 1:  #check if the current distance is smaller than the previous one
					distances[neigh] = distances[current] + 1
					parents[neigh] = current
	return found_target, distances[tuple(current)], np.array(current)

def rotate_features(features, n):
	if n==1: return features[[1,2,3,0,5,6,7,4,9,10,11,8,13,14,15,12,17,18,19,16,21,22,23,20,24,25]]
	if n==2: return rotate_features(features[[1,2,3,0,5,6,7,4,9,10,11,8,13,14,15,12,17,18,19,16,21,22,23,20,24,25]], 1)
	if n==3: return rotate_features(rotate_features(features[[1,2,3,0,5,6,7,4,9,10,11,8,13,14,15,12,17,18,19,16,21,22,23,20,24,25]], 1), 1)

def rotate_action(action, n):
	'''
	rotate action by n*90° counter clockwise
	'''
	if ACTIONS_DICT[action] in range(4):
		return ACTIONS[(ACTIONS_DICT[action] - n)%4]
	else:
		return action