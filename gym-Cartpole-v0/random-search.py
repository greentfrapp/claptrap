import gym
from gym import wrappers
import numpy as np

def main():
  env = gym.make('CartPole-v0')
  env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1',force=True)
  best_reward = 0
  best_param = None
  for _ in range(500):
    param = np.random.rand(4)*2-1
    reward = run_env(env,param)
    if reward>best_reward:
      best_param = param
      best_reward = reward
    if best_reward==200: break
  print best_reward

  for _ in range(100):
    run_env(env,best_param,no_of_steps=200)

  env.close()
  gym.upload('/tmp/cartpole-experiment-1', api_key='sk_hx1EEvQsTCOpk1lDufLDjw')


def run_env(env,param,no_of_steps=200,show=False):
  observation = env.reset()
  total_reward = 0
  for _ in range(no_of_steps):
    if show: env.render()
    if np.dot(param,observation)<0: action = 0
    else: action = 1
    observation, reward, done, info = env.step(action)
    total_reward += reward
    if done: break
  return total_reward


if __name__=="__main__": main()
