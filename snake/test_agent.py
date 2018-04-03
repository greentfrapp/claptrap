import numpy as np

from Snake import Snake

env = Snake()
obs = env.reset()
print(obs.grid)
print(obs.state)
print(obs.available_actions)
while obs.state == 1 or obs.state == 2:
	raw_input("ENTER")
	action = np.random.choice(obs.available_actions)
	print("ACTION : {}".format(action))
	obs = env.step(action)
	print(obs.grid)