import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

class Policy(object):

  def __init__(self):
    self.weights = {
      'w1':np.random.randn(2,4)
      }
    self.MS_grad = {
      'w1':np.zeros((2,4))
      }

  def get_action(self,state):
    output = np.matmul(self.weights['w1'],state)
    action_probabilities = softmax(output)
    action = 0 if np.random.uniform(0,1) < action_probabilities[0] else 1
    return action, action_probabilities

  def update(self,performance,history):

    # Calculate loss
    # where history.action_probs is tensor of probability vectors (nx2)
    # and history.actions is tensor of 1-hot action taken vectors (nx2)
    loss = float(-np.matmul(performance,np.log(np.sum(history.action_probs * history.actions, axis=1, keepdims=True))))

    # Calculate gradients
    gradients = {}
    gradients['w1'] = np.matmul(((history.actions - history.action_probs) * performance.T / len(performance)).T,history.states)

    # Update MS_grad
    beta = 0.9
    self.MS_grad['w1'] = beta * self.MS_grad['w1'] + (1-beta) * gradients['w1']**2

    # Finally, update weights with RMS_grad
    learning_rate = 0.01
    smoothing = 1e-8
    self.weights['w1'] = self.weights['w1'] - (learning_rate / np.sqrt(self.MS_grad['w1'] + smoothing)) * gradients['w1']

    return loss

class RewardModel(object):

  def __init__(self):
    # Initialize weights
    self.weights = {
      'w1':np.random.randn(10,4) / np.sqrt(4),
      'b1':np.zeros((10,1)),
      'w2':np.random.randn(1,10) / np.sqrt(10),
      'b2':np.zeros((1,1))
      }
    self.MS_grad = {
      'w1':np.zeros((10,4)),
      'b1':np.zeros((10,1)),
      'w2':np.zeros((1,10)),
      'b2':np.zeros((1,1))
      }
    self.hidden_layer_outputs = []

  def predict_cumulative_reward(self,history):
    # where history.states is nx4
    # it's okay to just use np.matmul(weights,history.states.T)+bias due to broadcast
    # hidden_layer_output 10xn
    hidden_layer_output = relu(np.matmul(self.weights['w1'],history.states.T)+self.weights['b1'])
    cumulative_reward = np.matmul(self.weights['w2'],hidden_layer_output)+self.weights['b2']
    
    self.hidden_layer_outputs = hidden_layer_output

    return cumulative_reward

  def update(self,predicted_cumulative_reward,performance,history):

    # performance.shape (1xn)
    loss = 0.5 * np.sum(performance**2)

    # Calculate gradients
    gradients = {}
    gradients['w2'] = (np.sum(self.hidden_layer_outputs * performance, axis=1) / len(performance)).reshape(1,10)
    delta_w2 = np.ones((1,10)) * np.mean(performance) / len(performance)
    gradients['b2'] = np.mean(performance).reshape(1,1)
    # 10x1 * 10x1 * 10xn . nx4 * nx1 * sum(1x10 * 1x10)
    gradients['w1'] = np.matmul(delta_w2.T * self.weights['w2'].T * heaviside(self.hidden_layer_outputs),history.states)
    gradients['b1'] = np.sum(delta_w2.T * self.weights['w2'].T * heaviside(self.hidden_layer_outputs),axis=1).reshape(10,1)
    
    # Update MS_grad
    beta = 0.9
    for key in self.MS_grad:
      self.MS_grad[key] = beta * self.MS_grad[key] + (1 - beta) * gradients[key]**2
    
    # Finally, update weights with RMS_grad
    learning_rate = 0.001
    smoothing = 1e-8
    for key in self.weights:
      self.weights[key] = -(learning_rate / np.sqrt(self.MS_grad[key] + smoothing)) * gradients[key]
    
    return loss

class History(object):

  def __init__(self):
    self.states = None
    self.actions = None
    self.action_probs = None
    self.rewards = None
    self.no_of_steps = 0

  def update(self,step):
    if self.no_of_steps == 0: 
      self.states = np.array([step['prev_state']])
      self.actions = np.array([step['action']])
      self.action_probs = np.array([step['action_prob']])
      self.rewards = np.array([step['reward']])
    else:
      self.states = np.append(self.states,[step['prev_state']],axis=0)
      self.actions = np.append(self.actions,[step['action']],axis=0)
      self.action_probs = np.append(self.action_probs,[step['action_prob']],axis=0)
      self.rewards = np.append(self.rewards,[step['reward']],axis=0)
    self.no_of_steps += 1

def main():
  env = gym.make('CartPole-v0')
  log_directory = '/tmp/cartpole-policy-gradient'
  #env = wrappers.Monitor(env,log_directory,force=True)

  reward,model_loss,policy_loss = [],[],[]

  policy = Policy()
  reward_model = RewardModel()

  # Run at most 5000 episodes for training
  for _ in range(1000):
    episode_reward,episode_model_loss,episode_policy_loss = run_episode(env,reward_model,policy)
    reward.append(episode_reward)
    model_loss.append(episode_model_loss)
    policy_loss.append(episode_policy_loss)

    #if episode_reward == 200: break

  # Run 100 more episodes for proving that exercise is solved
  for _ in range(100):
    run_episode(env,reward_model,policy)

  env.close()

  ax = plt.axes()
  ax.plot(range(len(reward)),reward,color='red')
  #ax.plot(range(len(model_loss)),model_loss,color='blue')
  #ax.plot(range(len(policy_loss)),policy_loss,color='green')

  plt.show()

def run_episode(env,reward_model,policy,no_of_steps=200):

  history = History()
  total_reward = 0

  state = env.reset()

  for _ in range(no_of_steps):
    #env.render()
    prev_state = state
    # Get action probabilities, select and run action
    action, action_prob = policy.get_action(state)
    state, reward, done, info = env.step(action)

    if action == 0: action_vector = [1,0]
    elif action == 1: action_vector = [0,1]

    history.update({'prev_state':prev_state,'action':action_vector,'action_prob':action_prob,'reward':reward})
    total_reward += reward

    if done: break

  # Calculate the true and predicted cumulative rewards
  true_cumulative_reward = get_cumulative_reward(history)
  predicted_cumulative_reward = reward_model.predict_cumulative_reward(history)

  # Calculate performance of the policy
  performance = predicted_cumulative_reward - true_cumulative_reward

  # Use predicted_cumulative_reward, performance and history to update reward_model
  model_loss = reward_model.update(predicted_cumulative_reward,performance,history)
  #model_loss = 0

  # Use performance and history to update policy
  policy_loss = policy.update(performance,history)
  #policy_loss = 0

  print str(total_reward)+" ModelLoss: "+str(model_loss)+" PolicyLoss: "+str(policy_loss)

  return total_reward,model_loss,policy_loss

def relu(vector):
  vector[vector < 0] = 0
  return vector

def heaviside(vector):
  vector = 0.5 * (np.sign(vector) + 1)
  return vector

def softmax(vector):
  vector = np.exp(vector)
  vector /= np.sum(vector)
  return vector

def get_cumulative_reward(history):
  cumulative_reward_history = np.array([])
  for idx in range(history.no_of_steps):
    cumulative_reward = 0
    discount = 1.0
    discount_decrement = 0.95
    for reward in history.rewards[idx:]:
      cumulative_reward += reward * discount
      discount *= discount_decrement
    cumulative_reward_history = np.append(cumulative_reward_history,cumulative_reward)
  return cumulative_reward_history

if __name__=="__main__": main()
