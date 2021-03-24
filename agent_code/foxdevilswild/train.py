import os
import pickle
import random
import numpy as np
from typing import List

import events as e
from .callbacks import state_to_features, regression_model

# Hyper parameters
GAMMA = 0 #discount factor
LEARNING_RATE = 1e-1
NUMBER_OF_TRAINING_INSTANCES = 3000
EPS_MIN = 0.05
EPS_MAX = 1
EPS_DEC = EPS_MIN**(1/(0.9*NUMBER_OF_TRAINING_INSTANCES))

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_DICT = {None:-1,'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}

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
		self.stats = np.array([0.0, 0.0, 0.0, 0.0]) #instance, rounds survived, coins collected, steps taken
		np.save('stats.npy', self.stats)
		self.instance = 0
		self.eps = EPS_MAX
	self.rounds_survived = 0
	self.coins_collected = 0
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
	
	if old_game_state != None and new_game_state != None:
		#approached coin?
		self_old = old_game_state['self'][3]
		self_new = new_game_state['self'][3]
		coins = old_game_state['coins']
		walls = old_game_state['field'].copy()
		walls[walls != -1] = 0
		dist_old = dist_to_nearest(tuple(self_old), walls, coins)
		dist_new = dist_to_nearest(tuple(self_new), walls, coins)
		if dist_old > dist_new: 
			events.append("APPROACHED_COIN")
		#approached crate?
		crate_walls = old_game_state['field'].copy()
		crate_pos = np.array(np.where(crate_walls == 1)).T
		crate_walls[crate_walls == 1] = 0
		dist_old = dist_to_nearest(tuple(self_old), crate_walls, crate_pos)
		dist_new = dist_to_nearest(tuple(self_new), crate_walls, crate_pos)
		if dist_old > dist_new: events.append("APPROACHED_CRATE")
		#escaping death?
		bombs = old_game_state['bombs']
		if len(bombs) > 0: 
			board = old_game_state['field']
			nogo = board.copy()
			oob = lambda x: not ( (np.array(x) >= 0).all() and (np.array(x) < 17).all() ) #out of bounds
			for b in bombs:
				nogo[tuple(b[0])] = 7
				stop_up = False; stop_right = False; stop_down = False; stop_left = False
				for i in range(1,5):
					t_up    = tuple(b[0] + i*np.array([ 0,-1]))
					t_right = tuple(b[0] + i*np.array([ 1, 0]))
					t_down  = tuple(b[0] + i*np.array([ 0, 1]))
					t_left  = tuple(b[0] + i*np.array([-1, 0]))
					if oob(t_up   ) or board[t_up   ] == -1: stop_up    = True
					if oob(t_right) or board[t_right] == -1: stop_right = True
					if oob(t_down ) or board[t_down ] == -1: stop_down  = True
					if oob(t_left ) or board[t_left ] == -1: stop_left  = True
					if not stop_up   : nogo[t_up   ] = 7
					if not stop_right: nogo[t_right] = 7
					if not stop_down : nogo[t_down ] = 7
					if not stop_left : nogo[t_left ] = 7
			#calculate coordinates of free tiles
			free_pos = np.array(np.where(nogo == 0)).T
			#calculate which way to go to
			if nogo[tuple(self_old)] == 7 and dist_to_nearest(tuple(self_old), board, free_pos) > dist_to_nearest(tuple(self_new), board, free_pos):
				events.append("ESCAPING_DEATH")
			#moved into death zone?
			if nogo[tuple(self_old)] == 0 and nogo[tuple(self_new)] == 7:
				events.append("MOVED_TO_DEATH")
	#placed bomb in a good spot?
	if "BOMB_DROPPED" in events:
		explosion_pos = np.array(new_game_state['self'][3])
		bomb_pos = new_game_state['self'][3]
		board = new_game_state['field']
		oob = lambda x: not ( (np.array(x) >= 0).all() and (np.array(x) < 17).all() ) #out of bounds
		stop_up = False; stop_right = False; stop_down = False; stop_left = False
		for i in range(1,5):
			t_up    = tuple(bomb_pos + i*np.array([ 0,-1]))
			t_right = tuple(bomb_pos + i*np.array([ 1, 0]))
			t_down  = tuple(bomb_pos + i*np.array([ 0, 1]))
			t_left  = tuple(bomb_pos + i*np.array([-1, 0]))
			if oob(t_up   ) or board[t_up   ] == -1: stop_up    = True
			if oob(t_right) or board[t_right] == -1: stop_right = True
			if oob(t_down ) or board[t_down ] == -1: stop_down  = True
			if oob(t_left ) or board[t_left ] == -1: stop_left  = True
			if not stop_up   : explosion_pos = np.vstack((explosion_pos, t_up   ))
			if not stop_right: explosion_pos = np.vstack((explosion_pos, t_right))
			if not stop_down : explosion_pos = np.vstack((explosion_pos, t_down ))
			if not stop_left : explosion_pos = np.vstack((explosion_pos, t_left ))
		counter = 0
		if len(new_game_state['others']) > 0: 
			others = new_game_state['others'].reshape((-1,4))[:,3]
			board[(others[:,0], others[:,1])] = 1
		for e in explosion_pos:
			if board[tuple(e)] == 1: counter += 1
		if counter > 0: events.append("WELL_BOMBED")
		for i in range(1,counter): events.append("ADDITIONAL_BOMBAGE")
	
	#self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
	
	#we have to use self.last_action for the action otherwise invalid actions wont get penalized
	if old_game_state != None and new_game_state != None:
		self.transitions.append([state_to_features(old_game_state), self.last_action, state_to_features(new_game_state), reward_from_events(self, events)])
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
	self.steps_taken = last_game_state['step']
	self.stats = np.vstack((self.stats, [self.instance, self.rounds_survived, self.coins_collected, self.steps_taken]))
	np.save('stats.npy', self.stats)
	self.coins_collected = 0
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
		"MOVED_LEFT"			: -5,
		"MOVED_RIGHT"			: -5,
		"MOVED_UP"				: -5,
		"MOVED_DOWN"			: -5,

		"WAITED"				: -5,
		"INVALID_ACTION"		: -5,

		"APPROACHED_CRATE"		: 15,
		"BOMB_DROPPED"			: -1,
		"WELL_BOMBED"			: 2,
		"ADDITIONAL_BOMBAGE"	: 1,
		
		"BOMB_EXPLODED"			: 0,
		"CRATE_DESTROYED"		: 0,
		"COIN_FOUND"			: 0,

		"APPROACHED_COIN"		: 35,
		"COIN_COLLECTED"		: 35,

		"KILLED_OPPONENT"		: 0,

		"MOVED_TO_DEATH"		: -40,
		"KILLED_SELF"			: 0,
		"GOT_KILLED"			: 0,

		"OPPONENT_ELIMINATED"	: 0,

		"ESCAPING_DEATH"		: 65,
		"SURVIVED_ROUND" 		: 0
	}
	reward_sum = 0
	for event in events:
		if event in game_rewards:
			reward_sum += game_rewards[event]
	self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
	return reward_sum

def q_learning(self, prev_states, next_states, rewards, actions):
	T = len(rewards)
	ans = np.zeros(T)
	for t in range(T):
		old_q = regression_model(self, prev_states[t], action=actions[t])
		max_q = regression_model(self, next_states[t]).max()
		ans[t] = old_q + LEARNING_RATE*(rewards[t] + GAMMA*max_q - old_q)
		return ans

def n_step_td_learning(self, rewards, next_states_features, n_steps=1):
	T = len(rewards)
	y = np.zeros(T)
	for t in range(T):
		if T-t-1<n_steps: n_steps = T-t-1
		y[t] = np.sum((GAMMA**np.arange(n_steps))*rewards[t+1:t+1+n_steps]) + (GAMMA**n_steps)*regression_model(self, next_states_features).max()
	return y

def td_learning(self, rewards, next_states_features):
	T = len(rewards)
	y = np.zeros(T)
	for t in range(T):
		y[t] = rewards[t] + GAMMA*regression_model(self, next_states_features).max()
	return y

def mc(rewards):
	#monte carlo value approximation
	T = len(rewards)
	y = np.zeros(T)
	for i in range(T):
		#discounted return for full episode
		y[i] = np.sum(GAMMA**np.arange(T-i)*rewards[i:])
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

def dist_to_nearest(start: tuple, wall_map: np.array, coin_pos: np.array):
	'''
	finds the best way to go by applying dyjkstras algorithm
	:start: tuple of starting coordinates
	:wall_map: np.array of the whole board with 0 where there is a free tile
	:coin_pos: np.array of coordinates of target positions
	:ans: np.array of features (1 if this way is the ideal way, 0 if not)
	'''
	#board dimensions
	dim = wall_map.shape
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
	
	#if there arent any coins return 0
	if len(coin_pos) == 0: return 0
	
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
		if isin(current, coin_pos):
			#final = current
			break
		#now give all the neighbors a distance based on the distance of the current tile
		for t in translations:
			neigh = (current[0]+t[0], current[1]+t[1])
			if within_bounds(neigh) and wall_map[neigh] == 0: #check if that is a free tile
				if neigh not in inspecting and neigh not in done: inspecting.append(neigh) #check if this tile hasnt been used yet
				if neigh not in distances or distances[neigh] > distances[current] + 1:  #check if the current distance is smaller than the previous one
					distances[neigh] = distances[current] + 1
					parents[neigh] = current
	return distances[tuple(current)]