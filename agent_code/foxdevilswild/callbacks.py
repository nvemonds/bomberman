import os
import pickle
import random

import numpy as np

#Hyperparameter
TEMPERATURE = 1e1
FEATURE_DIMS = 26 #not a hyperparameter but dont wanna hard code it

#Helpful stuff
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
ACTIONS_DICT = {None:-1,'UP':0, 'RIGHT':1, 'DOWN':2, 'LEFT':3, 'WAIT':4, 'BOMB':5}


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
	if os.path.isfile("my-saved-model.pt"): #if has been trained before load model
		self.logger.info("Loading model from saved state.")
		with open("my-saved-model.pt", "rb") as file:
			self.model = pickle.load(file)
	else: #if not initialize with identical betas
		self.logger.info("Setting up model from scratch.")
		self.model = np.zeros((len(ACTIONS),FEATURE_DIMS))
		#self.model = np.array([[25,0,0,0,15,0,0,0,-15,0,0,0,-30,0,0,0,35,0,0,0,0,-5],
		#					   [0,25,0,0,0,15,0,0,0,-15,0,0,0,-30,0,0,0,35,0,0,0,-5],
		#					   [0,0,25,0,0,0,15,0,0,0,-15,0,0,0,-30,0,0,0,35,0,0,-5],
		#					   [0,0,0,25,0,0,0,15,0,0,0,-15,0,0,0,-30,0,0,0,35,0,-5],
		#					   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-5],
		#					   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1]], dtype=np.float32)
	return

def act(self, game_state: dict) -> str:
	"""
	Your agent should parse the input, think, and take a decision.
	When not in training mode, the maximum execution time for this method is 0.5s.

	:param self: The same object that is passed to all of your callbacks.
	:param game_state: The dictionary that describes everything on the board.
	:return: The action to take as a string.
	"""
	#go through all possible actions and find their Q function value
	values = regression_model(self, state_to_features(game_state))
	self.logger.debug(f'Features are {state_to_features(game_state)}')
	self.logger.debug(f'Action-values are {values}')
	if self.train and random.random() <= self.eps:
		#calculate the Q function value for all actions and execute according to a softmax distribution
		#prob = np.exp(values/TEMPERATURE)/np.sum(np.exp(values/TEMPERATURE))
		prob = 1/len(ACTIONS)*np.ones(len(ACTIONS)) #uniform
	else:
		#if not training execute action with highest action-value
		prob = np.zeros(len(ACTIONS))
		prob[np.argmax(values)] = 1
	self.logger.debug(f'Probabilites for all Actions are { {a:p for a,p in zip(ACTIONS, prob) } }')
	
	#save the executed action
	self.last_action = np.random.choice(ACTIONS, p=prob)
	
	self.logger.debug(f'Executing Action {self.last_action}')
	
	#return action
	return self.last_action

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
	# This is the dict before the game begins and after it ends
	if game_state is None: return None
	
	#extract own position
	self_pos = np.array(tuple(game_state['self'][3]))
	#translation unit vectors
	left  = np.array([-1, 0])
	right = np.array([ 1, 0])
	up    = np.array([ 0,-1])
	down  = np.array([ 0, 1])
	#list of neighboring tiles
	adj_tiles = np.array([self_pos + up, self_pos + right, self_pos + down, self_pos + left])

	#extract coin and wall positions and such...
	coin_pos = np.array(game_state['coins'])		#coin positions
	board = game_state['field']						#wall and crate map
	walls = board.copy()
	walls[walls != -1] = 0							#wall map
	crate_pos = np.array(np.where(board == 1)).T	#crate positions
	bombs = game_state['bombs']						#bomb pos and ticks
	others = game_state['others']
	if len(others) > 0:
		others_pos = np.empty((len(others),2), dtype=int)
		for i in range(len(others)): others_pos[i] = others[i][3]
	non_passable_tiles = board.copy()
	if len(others) > 0: non_passable_tiles[(others_pos[:,0], others_pos[:,1])] = 1

	#if within the explosion radius of a bomb determine which way to go to escape it
	u_f, r_f, d_f, l_f = (0, 0, 0, 0)	#f for free tile
	nogo = board.copy()		#nogo will be map of walls (-1), crates (1) and 'death' tiles (7) (aka tiles within the explosion radius of a bomb)
	if len(bombs) > 0:
		#go through all bombs and calculate their explosion radii
		for b in bombs:
			explosions = explosion_radius(b[0], board)
			nogo[(explosions[:,0], explosions[:,1])] = 7
		if nogo[tuple(self_pos)] == 7: #if within the explosion radius of a bomb
			#calculate coordinates of free tiles
			free_pos = np.array(np.where(nogo == 0)).T
			#calculate which way to go to
			u_f, r_f, d_f, l_f, _ = find_the_way(tuple(self_pos), board, free_pos)

	#find the best tile to get to the next coin or crate
	u_c , r_c , d_c , l_c, _ = find_the_way(tuple(self_pos), non_passable_tiles, coin_pos)	#c for coin
	
	#which way to go to get to the nearest crate
	crate_borders = non_passable_tiles.copy()	
	crate_borders[board == 1] = 0 #crates arent borders to get to crates
	u_cr, r_cr, d_cr, l_cr, cr_dist = find_the_way(tuple(self_pos), crate_borders, crate_pos)
	if cr_dist == 1: u_cr, r_cr, d_cr, l_cr = (0,0,0,0) 	#if distance is 1 then dont go there cuz its gon be invalid
	
	#check if there are walls or crates on any of the adjacent tiles (aka does moving there result in an invalid move?)
	u_w = np.abs(non_passable_tiles[tuple(adj_tiles[0])])
	r_w = np.abs(non_passable_tiles[tuple(adj_tiles[1])])
	d_w = np.abs(non_passable_tiles[tuple(adj_tiles[2])])
	l_w = np.abs(non_passable_tiles[tuple(adj_tiles[3])])
	
	#check if any of the neighboring tiles are death tiles (if not already within the explosion radius of a bomb)
	death = np.zeros(board.shape)
	death[nogo == 7] = 1
	if death[tuple(self_pos)] == 0:
		u_d = death[tuple(self_pos + up   )]
		r_d = death[tuple(self_pos + right)]
		d_d = death[tuple(self_pos + down )]
		l_d = death[tuple(self_pos + left )]
	else:
		u_d, r_d, d_d, l_d = (0,0,0,0)
	
	#additional feature to determine if a bomb should be used
	to_bomb_or_not_to_bomb = 0
	if game_state['self'][2] == 1:
		potential_explosions = explosion_radius(self_pos, board)
		for d in potential_explosions:
			if board[tuple(d)] == 1: 
				to_bomb_or_not_to_bomb = 1
				break
			others = game_state['others']
			if len(others) > 0:
				if np.array(others, dtype=object).ndim == 1 and (others[3] == d).all(): #if only one enemy agent
					to_bomb_or_not_to_bomb = 1
				else:											#if multiple enemy agents
					for o in others:
						if (np.array(o[3]) == d).all(): 
							to_bomb_or_not_to_bomb = 1
							break
	
	#check which way to go to approach other agents
	u_o, r_o, d_o, l_o = (0,0,0,0)
	if len(others) > 0:
		u_o, r_o, d_o, l_o, _ = find_the_way(tuple(self_pos), board, others_pos)

	return np.array([u_c, r_c, d_c, l_c, u_cr, r_cr, d_cr, l_cr, u_o, r_o, d_o, l_o, u_w, r_w, d_w, l_w, u_d, r_d, d_d, l_d, u_f, r_f, d_f, l_f, to_bomb_or_not_to_bomb, 1])

def explosion_radius(bomb_pos, obstructions):
	explosions = np.array([bomb_pos])
	oob = lambda x: not ( (np.array(x) >= 0).all() and (np.array(x) < 17).all() ) #out of bounds?
	stop_up = False; stop_right = False; stop_down = False; stop_left = False
	for i in range(1,5):
		t_up    = tuple(bomb_pos + i*np.array([ 0,-1]))
		t_right = tuple(bomb_pos + i*np.array([ 1, 0]))
		t_down  = tuple(bomb_pos + i*np.array([ 0, 1]))
		t_left  = tuple(bomb_pos + i*np.array([-1, 0]))
		if oob(t_up   ) or obstructions[t_up   ] == -1: stop_up    = True
		if oob(t_right) or obstructions[t_right] == -1: stop_right = True
		if oob(t_down ) or obstructions[t_down ] == -1: stop_down  = True
		if oob(t_left ) or obstructions[t_left ] == -1: stop_left  = True
		if not stop_up   : explosions = np.vstack((explosions, t_up   ))
		if not stop_right: explosions = np.vstack((explosions, t_right))
		if not stop_down : explosions = np.vstack((explosions, t_down ))
		if not stop_left : explosions = np.vstack((explosions, t_left ))
	return explosions

def find_the_way(start: tuple, wall_map: np.array, coin_pos: np.array, max_dist = 100):
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
	
	#initialize all four features as 0, the one that ist best to go along will be set to 1 then
	ans = np.zeros(5) #up, right, down, left, dist
	
	#some auxiliary stacks and dicts
	inspecting = [start]
	done = []
	parents = {start:start}
	distances = {start: 0}
	translations = [(0,-1), (1,0), (0,1), (-1,0)] #up, right, down, left
	found = False
	
	#if there arent any coins return 0 0 0 0
	if len(coin_pos) == 0: return ans
	
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
			found = True
			break

		#else give all the neighbors a distance based on the distance of the current tile
		for t in translations:
			neigh = (current[0]+t[0], current[1]+t[1])
			if within_bounds(neigh) and wall_map[neigh] == 0 and distances[current] < max_dist: #check if that is a free tile and the max distance hasnt been exceeded yet
				if neigh not in inspecting and neigh not in done: inspecting.append(neigh) #check if this tile hasnt been used yet
				if neigh not in distances or distances[neigh] > distances[current] + 1:  #check if the current distance is smaller than the previous one
					distances[neigh] = distances[current] + 1
					parents[neigh] = current
	ans[4] = distances[tuple(current)]
	#if there is no way to any of the targets
	if not found:
		return ans
	#trace back the path to the coin and stop one tile before the starting tile so we know which way to go
	while parents[current] != start:
		current = parents[current]
	#find the right move that gets you to the target
	for i in range(4):
		t = translations[i]
		if current == (start[0] + t[0], start[1] + t[1]):
			ans[i] = 1
			break
	#check if going any other way would get you to a target with the same distance
	u, r, d, l, dist = tuple(ans)
	new_map = wall_map.copy()
	while (np.array([u, r, d, l]) == 1).any() and max_dist == 100: #max recursion depth is 1
		t = translations[np.where(np.array([u, r, d, l]) == 1)[0][0]]
		coord = (start[0]+t[0], start[1]+t[1])
		new_map[coord] = -1
		u, r, d, l, dist = find_the_way(start, new_map, coin_pos, max_dist=ans[4])
		if dist == ans[4] and (np.array([u, r, d, l]) == 1).any(): ans += np.array([u, r, d, l, 0])
	return tuple(ans)

def regression_model(self, features, action=''):
	"""
	linear regression model
		returns the approximate Q value for a given action 
		or for each possible action if action=''
	"""
	if action != '':
		ans = features@self.model[ACTIONS_DICT[action]]
	else:
		ans = np.empty(len(ACTIONS))
		for i in range(len(ACTIONS)):
			ans[i] = features@self.model[i]
	return ans