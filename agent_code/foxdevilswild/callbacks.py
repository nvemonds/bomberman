import os
import pickle
import random

import numpy as np

#Hyperparameter
TEMPERATURE = 1e1
FEATURE_DIMS = 18 #not a hyperparameter but dont wanna hard code it

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
		#self.model = np.array([[10,0,0,0,-10],
		#					   [0,10,0,0,-10],
		#					   [0,0,10,0,-10],
		#					   [0,0,0,10,-10],
		#					   [0,0,0,0,0],
		#					   [-10,-10,-10,-10,100]], dtype=np.float32)
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
	left  = np.array([-1, 0])
	right = np.array([ 1, 0])
	up    = np.array([ 0,-1])
	down  = np.array([ 0, 1])
	adj_tiles = np.array([self_pos + up, self_pos + right, self_pos + down, self_pos + left])

	#extract coin and wall positions
	coin_pos = np.array(game_state['coins'])
	board = game_state['field']
	walls = board.copy()
	walls[walls != -1] = 0
	crate_pos = np.array(np.where(board == 1)).T

	#extract bomb positions and create a map of 'nogo-zones'
	nogo = board.copy()
	bombs = game_state['bombs']
	if len(bombs) > 0: 
		oob = lambda x: not ( (np.array(x) >= 0).all() and (np.array(x) < 17).all() ) #out of bounds
		for b in bombs:
			nogo[tuple(b[0])] = 7
			stop_up = False; stop_right = False; stop_down = False; stop_left = False
			for i in range(1,5):
				t_up    = tuple(b[0] + i*up   )
				t_right = tuple(b[0] + i*right)
				t_down  = tuple(b[0] + i*down )
				t_left  = tuple(b[0] + i*left )
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
		f_u, f_r, f_d, f_l = (0, 0, 0, 0)
		if nogo[tuple(self_pos)] == 7: f_u, f_r, f_d, f_l, _ = find_the_way(tuple(self_pos), board, free_pos)
	else:
		f_u, f_r, f_d, f_l = (0, 0, 0, 0)

	#find the best tile to get to the next coin or crate
	u_c , r_c , d_c , l_c, _ = find_the_way(tuple(self_pos), nogo, coin_pos )
	#if within a death zone, escaping that is more important than collecting coins
	if (np.array([f_u, f_r, f_d, f_l]) != 0).any():
		u_c = 2*f_u; r_c = 2*f_r; d_c = 2*f_d; l_c = 2*f_l
	crate_borders = nogo.copy()
	crate_borders[nogo == 1] = 0
	u_cr, r_cr, d_cr, l_cr, cr_dist = find_the_way(tuple(self_pos), crate_borders, crate_pos)
	if cr_dist == 1: u_cr, r_cr, d_cr, l_cr = (0,0,0,0) #if distance is 1 then dont go there cuz its gon be invalid
	
	#check if there are walls or on any of the adjacent tiles (aka does moving there result in an invalid move?)
	u_w = np.abs(np.min([0, board[tuple(adj_tiles[0])]]))
	r_w = np.abs(np.min([0, board[tuple(adj_tiles[1])]]))
	d_w = np.abs(np.min([0, board[tuple(adj_tiles[2])]]))
	l_w = np.abs(np.min([0, board[tuple(adj_tiles[3])]]))
	
	#check if any of the neighboring tiles are death tiles
	death = np.zeros(board.shape)
	death[nogo == 7] = 1
	u_d = death[tuple(self_pos + up   )]
	r_d = death[tuple(self_pos + right)]
	d_d = death[tuple(self_pos + down )]
	l_d = death[tuple(self_pos + left )]
	
	#additional feature to determine if a bomb should be used (and how badly it should be used)
	to_bomb_or_not_to_bomb = 0
	if game_state['self'][2] == 1:
		death_pos = np.array(np.where(death==1)).T
		for d in death_pos:
			if board[tuple(death_pos)] == 1: to_bomb_or_not_to_bomb += 1
			for o in game_state['others'][:,3]:
				if (o == d).all(): to_bomb_or_not_to_bomb += 1
	# 1: use a bomb
	#-1: dont
	
	#print("total features: ", u, r, d, l, to_bomb_or_not_to_bomb)
	return np.array([u_c, r_c, d_c, l_c, u_cr, r_cr, d_cr, l_cr, u_w, r_w, d_w, l_w, u_d, r_d, d_d, l_d, to_bomb_or_not_to_bomb, 1])

def find_the_way(start: tuple, wall_map: np.array, coin_pos: np.array):
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
			break
		#now give all the neighbors a distance based on the distance of the current tile
		for t in translations:
			neigh = (current[0]+t[0], current[1]+t[1])
			if within_bounds(neigh) and wall_map[neigh] == 0: #check if that is a free tile
				if neigh not in inspecting and neigh not in done: inspecting.append(neigh) #check if this tile hasnt been used yet
				if neigh not in distances or distances[neigh] > distances[current] + 1:  #check if the current distance is smaller than the previous one
					distances[neigh] = distances[current] + 1
					parents[neigh] = current
	#trace back the path to the coin and stop one tile before the starting tile so we know which way to go
	#current = final
	#print(f'found nearest: {current}')
	ans[4] = distances[tuple(current)]
	if not isin(current, coin_pos): #if there is no unlocked way to a coin
		return ans
	while parents[current] != start:
		current = parents[current]
	#print(f'its parent is: {current}')
	#way to go is
	for i in range(4):
		t = translations[i]
		if current == (start[0] + t[0], start[1] + t[1]):
			ans[i] = 1
			break
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
		ans = np.array([])
		for beta in self.model:
			ans = np.append(ans, features@beta)
	return ans