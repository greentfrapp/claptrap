import numpy as np

from Snake import Snake

env = Snake(10,10)
obs = env.reset()
print(obs.grid)
while obs.state == 1 or obs.state == 2:
	action = raw_input("ACTION: ")
	if action == 'w':
		action = 1
	elif action == 'd':
		action = 2
	elif action == 's':
		action = 3
	elif action == 'a':
		action = 4
	elif action == '':
		action = 0
	else:
		quit()
	obs = env.step(action)
	print(obs.grid)
	print(obs.score)
	print(env.timestep)