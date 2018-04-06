import numpy as np

class Action(object):
	def __init__(self, action_id, action_name):
		self.id = action_id
		self.name = action_name

class Observation(object):
	def __init__(self, grid, score=0, reward=0, available_actions=[], state=0):
		self.grid = grid
		self.score = score
		self.reward = reward
		self.available_actions = available_actions
		self.state = state

class Snake(object):
	def __init__(self, grid_height=10, grid_width=10):
		self.max_y = grid_height - 1
		self.max_x = grid_width - 1
		self.reset_grid()
		self.food = []
		self.snake = []
		self.init_length = 3
		
		# Internally track current snake direction
		# 0 - UP, 1 - LEFT, 2 - DOWN, 3 - RIGHT
		self.current_direction = 0
		self.UP = 1
		self.RIGHT = 2
		self.DOWN = 3
		self.LEFT = 4
		
		self.score = 0
		self.reward = 0
		self.timestep = 0
		self.MAX_TIMESTEP = 1000

		# Track state
		# 0 - INACTIVE, 1 - START, 2 - RUNNING, 3 - END
		self.state = 0
		self.STATES = ["INACTIVE", "START", "RUNNING", "END"]
		self.STATE_INACTIVE = 0
		self.STATE_START = 1
		self.STATE_RUNNING = 2
		self.STATE_END = 3
		self.init_actions()

	def reset_grid(self):
		self.grid = np.zeros((self.max_y+1, self.max_x+1))

	def init_actions(self):
		self.ALL_ACTIONS = [Action(0, "NO_OP"), Action(1, "UP"), Action(2, "RIGHT"), Action(3, "DOWN"), Action(4, "LEFT")]
		self.available_actions = []

	def reset(self):
		assert self.state == self.STATE_INACTIVE or self.state == self.STATE_END, "Cannot run reset method when state is in {}:{}, run step instead".format(self.state, self.STATES[self.state])
		self.state = self.STATE_START
		# Create snake and feed snake
		self.genesis()
		self.score = 0
		self.reward = 0
		self.timestep = 0
		# Return Observation
		return Observation(self.grid, self.score, self.reward, self.available_actions, self.state)

	def step(self, action_id):
		if self.timestep == self.MAX_TIMESTEP:
			self.die()
			return Observation(self.grid, self.score, self.reward, self.available_actions, self.state)
		self.timestep += 1
		assert self.state == self.STATE_START or self.state == self.STATE_RUNNING, "Cannot run step method when state is in {}:{}, run reset instead".format(self.state, self.STATES[self.state])
		if self.state == self.STATE_START:
			self.state = self.STATE_RUNNING
		assert action_id in self.available_actions, "Action {}:{} not in list of available actions".format(self.ALL_ACTIONS[action_id].id, self.ALL_ACTIONS[action_id].name)

		self.reward = 0

		# What happens to env due to action taken
		if self.ALL_ACTIONS[action_id].id != 0:
			self.current_direction = self.ALL_ACTIONS[action_id].id
		self.slither()
		self.update_grid()

		# Return Observation
		return Observation(self.grid, self.score, self.reward, self.available_actions, self.state)

	def slither(self):
		head = self.snake[-1]
		dx = 0
		dy = 0
		self.available_actions = np.arange(len(self.ALL_ACTIONS))
		if self.current_direction == self.UP:
			dy = -1
			self.available_actions = np.delete(self.available_actions, self.DOWN)
		elif self.current_direction == self.LEFT:
			dx = -1
			self.available_actions = np.delete(self.available_actions, self.RIGHT)
		elif self.current_direction == self.DOWN:
			dy = 1
			self.available_actions = np.delete(self.available_actions, self.UP)
		elif self.current_direction == self.RIGHT:
			dx = 1
			self.available_actions = np.delete(self.available_actions, self.LEFT)
		new_head = head + [dy, dx]
		if self.collide(new_head):
			self.die()
			return None
		self.snake = np.concatenate((self.snake, [new_head]))
		if np.array_equal(self.food, new_head):
			self.score += 1
			self.reward = 1
			self.feed()
		else:
			self.snake = self.snake[1:]

	def feed(self):
		x = np.random.choice(np.arange(self.max_x + 1))
		y = np.random.choice(np.arange(self.max_y + 1))
		while x in self.snake[:, 0] and y in self.snake[:, 1]:
			x = np.random.choice(np.arange(self.max_x + 1))
			y = np.random.choice(np.arange(self.max_y + 1))
		self.food = [x, y]

	def update_grid(self):
		self.reset_grid()
		for [x, y] in self.snake:
			self.grid[x, y] = 1
		self.grid[self.food[0], self.food[1]] = -1

	def genesis(self):
		self.reset_grid()
		self.snake = np.array([[0, 0]])
		while len(self.snake) < self.init_length:
			self.snake = np.concatenate((self.snake, [self.snake[-1] + [0, 1]]))
		self.current_direction = self.RIGHT
		self.available_actions = np.arange(len(self.ALL_ACTIONS))
		self.available_actions = np.delete(self.available_actions, self.LEFT)
		self.feed()
		self.update_grid()

	def collide(self, pos):
		if pos[0] < 0 or pos[0] > self.max_y:
			return True
		if pos[1] < 0 or pos[1] > self.max_x:
			return True
		for segment in self.snake:
			if np.array_equal(segment, pos):
				return True

	def die(self):
		self.state = self.STATE_END
