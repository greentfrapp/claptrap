import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

class Policy(object):

  def __init__(self):
    self.weights = {
      'w1':np.random.randn(2,4)
      }
    self.RMS_grad = {
      'w1':np.zeros((2,4))
      }

  def get_action(self,state):
    output = np.matmul(self.weights['w1'],state)
    action_probabilities = softmax(output)
    action = 0 if np.random.uniform(0,1) < action_probabilities[0] else 1
    return action, action_probabilities

  def update(self,performance,history):

    # Calculate loss
    # where action_prob is tensor of probability vectors (nx2)
    # and action is tensor of 1-hot action taken vectors (nx2)
    # Get actions as 1-hot vectors, action_prob and states from history
    action_prob = []
    action = []
    states = []
    for step in history:
      action_prob.append(step['action_prob'])
      states.append(step['prev_state'])
      if step['action'] == 0: action.append([1,0])
      else: action.append([0,1]) 
    action_prob = np.array(action_prob)
    actions = np.array(action)
    states = np.array(states)
    loss = float(-np.matmul(performance,np.log(np.sum(action_prob*actions,axis=1,keepdims=True))))

    # Calculate gradients
    grad_w1 = np.matmul(((actions - action_prob) * performance.T / len(performance)).T,states)

    # Update RMS_grad
    self.RMS_grad['w1'] = 0.9 * self.RMS_grad['w1'] + 0.1 * grad_w1**2

    # Finally, update weights with RMS_grad
    learning_rate = 0.01
    smoothing = 1e-8
    self.weights['w1'] = self.weights['w1'] - (learning_rate / np.sqrt(self.RMS_grad['w1'] + smoothing)) * grad_w1

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
    self.RMS_grad = {
      'w1':np.zeros((10,4)),
      'b1':np.zeros((10,1)),
      'w2':np.zeros((1,10)),
      'b2':np.zeros((1,1))
      }
    self.hidden_layer_outputs = []

  def predict_cumulative_reward(self,history):
    # where state is nx4
    # it's okay to just use np.matmul(weights,states.T)+bias due to broadcast
    # hidden_layer_output 10xn
    states = []
    for step in history: states.append(step['prev_state'])
    states = np.array(states).reshape(len(states),4)
    hidden_layer_output = relu(np.matmul(self.weights['w1'],states.T)+self.weights['b1'])
    cumulative_reward = np.matmul(self.weights['w2'],hidden_layer_output)+self.weights['b2']
    
    self.hidden_layer_outputs = hidden_layer_output

    return cumulative_reward

    


  def update(self,predicted_cumulative_reward,performance,history):

    # performance.shape (1xn)
    loss = 0.5 * np.sum(performance**2)

    # Get states from history
    states = []
    for step in history: states.append(step['prev_state'])
    states = np.array(states).reshape(len(states),4)

    # Calculate gradients

    grad_w2 = (np.sum(self.hidden_layer_outputs * performance, axis=1) / len(performance)).reshape(1,10)
    delta_w2 = np.ones((1,10)) * np.mean(performance) / len(performance)
    grad_b2 = np.mean(performance).reshape(1,1)
    # 10x1 * 10x1 * 10xn . nx4 * nx1 * sum(1x10 * 1x10)
    grad_w1 = np.matmul(delta_w2.T * self.weights['w2'].T * heaviside(self.hidden_layer_outputs),states)
    grad_b1 = np.sum(delta_w2.T * self.weights['w2'].T * heaviside(self.hidden_layer_outputs),axis=1).reshape(10,1)
    
    # Update RMS_grad
    self.RMS_grad['w1'] = 0.9 * self.RMS_grad['w1'] + 0.1 * grad_w1**2
    self.RMS_grad['b1'] = 0.9 * self.RMS_grad['b1'] + 0.1 * grad_b1**2
    self.RMS_grad['w2'] = 0.9 * self.RMS_grad['w2'] + 0.1 * grad_w2**2
    self.RMS_grad['b2'] = 0.9 * self.RMS_grad['b2'] + 0.1 * grad_b2**2

    # Finally, update weights with RMS_grad
    learning_rate = 0.001
    smoothing = 1e-8
    self.weights['w1'] = self.weights['w1'] - (learning_rate / np.sqrt(self.RMS_grad['w1'] + smoothing)) * grad_w1
    self.weights['b1'] = self.weights['b1'] - (learning_rate / np.sqrt(self.RMS_grad['b1'] + smoothing)) * grad_b1
    self.weights['w2'] = self.weights['w2'] - (learning_rate / np.sqrt(self.RMS_grad['w2'] + smoothing)) * grad_w2
    self.weights['b2'] = self.weights['b2'] - (learning_rate / np.sqrt(self.RMS_grad['b2'] + smoothing)) * grad_b2

    return loss

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

  history = []
  total_reward = 0

  state = env.reset()

  for _ in range(no_of_steps):
    #env.render()
    prev_state = state
    # Get action probabilities, select and run action
    action, action_prob = policy.get_action(state)
    state, reward, done, info = env.step(action)

    history.append({'prev_state':prev_state,'action':action,'action_prob':action_prob,'reward':reward})
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
  return np.array(cumulative_reward_history)

if __name__=="__main__": main()
