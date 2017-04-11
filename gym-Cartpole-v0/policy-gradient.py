import gym
from gym import wrappers
import numpy as np

class Policy(object):

  def __init__(self):
    w1 = np.random.randn(2,4)
    self.weights = {
      'w1':w1,
      }

  def get_action(self,state):
    output = np.matmul(self.weights['w1'],state)
    action_probabilities = softmax(output)
    action = 0 if random.uniform(0,1) < action_probabilities[0] else 1
    return action

  def update(self,performance,history):


class RewardModel(object):

  def __init__(self):
    # Initialize weights
    w1 = np.random.randn(10,4) / np.sqrt(4)
    w2 = np.random.randn(1,10) / np.sqrt(10)
    b1 = np.zeros((10,1))
    b2 = np.zeros((1,1))
    self.weights = {
      'w1':w1,
      'b1':b1,
      'w2':w2,
      'b2':b2
      }

  def predict_cumulative_reward(self,history):
    cumulative_reward_history = []
    for step in history:
      cumulative_reward = np.matmul(self.weights['w2'],relu(np.matmul(self.weights['w1'],step['prev_state'])+self.weights['b1']))+self.weights['b2']
      cumulative_reward_history.append(cumulative_reward)
    return cumulative_reward_history

  def update(self,true_cumulative_reward,history)

def main():
  env = gym.make('CartPole-v0')
  log_directory = '/tmp/cartpole-policy-gradient'
  #env = wrappers.Monitor(env,log_directory,force=True)

  policy = Policy()
  reward_model = RewardModel()

  # Run at most 1000 episodes for training
  for _ in range(1000):
    total_reward = run_episode(env,reward_model,policy)
    if total_reward == 200: break

  # Run 100 more episodes for proving that exercise is solved
  for _ in range(100):
    run_episode(env,reward_model,policy)

  env.close()

def run_episode(env,reward_model,policy,no_of_steps=200):

  history = []
  total_reward = 0

  state = env.reset()

  for _ in range(no_of_steps):
    prev_state = state
    # Get action probabilities, select and run action
    action = policy.get_action(state)
    state, reward, done, info = env.step(action)

    history.append({'prev_state':prev_state,'action':action,'reward':reward})
    total_reward += reward

    if done: break

  # Calculate the true and predicted cumulative rewards
  true_cumulative_reward = get_cumulative_reward(history)
  predicted_cumulative_reward = reward_model.predict_cumulative_reward(history)

  # Calculate performance of the policy
  performance = predicted_cumulative_reward - true_cumulative_reward

  # Use true_cumulative_reward and history to update reward_model
  reward_model.update(true_cumulative_reward,history)

  # Use performance and history to update policy
  policy.update(performance,history)

  return total_reward

def relu(vector):
  vector[vector < 0] = 0
  return vector

def softmax(vector):
  vector = np.exp(vector)
  vector /= np.sum(vector)
  return vector

def get_cumulative_reward(history):
  cumulative_reward_history = []
  for idx,step in enumerate(history):
    no_of_steps_left = len(history)-idx
    cumulative_reward = 0
    discount = 1.00
    discount_decrement = 0.95
    for idx2,step in enumerate(history[idx:]):
      cumulative_reward += step['reward'] * discount
      discount *= discount_decrement
    cumulative_reward_history.append(cumulative_reward)
  return np.array(cumulative_reward)

if __name__=="__main__": main()
