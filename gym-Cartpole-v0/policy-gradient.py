import gym
from gym import wrappers
import numpy as np

class Policy(object):

  def __init__(self):
    w1 = np.random.randn(2,4)
    self.weights = {
      'w1':w1,
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
    loss = 0
    for idx,step in enumerate(history):
      loss -= np.log(step['action_prob'][step['action']]) * performance[idx]

    # Get actions as 1-hot vectors, action_prob and states from history
    actions = []
    action_prob = []
    states = []
    for step in history:
      action_prob.append(step['action_prob'])
      states.append(step['prev_state'])
      if step['action'] == 0:
        actions.append([1,0])
      else:
        actions.append([0,1])

    actions = np.array(actions)
    action_prob = np.array(action_prob)
    states = np.array(states)

    # Calculate gradients
    grad_w1 = []
    for idx in range(2):
      grad_w1.append([])
      for idx2 in range(4):
        grad = 0
        for idx3,action in enumerate(actions):
          grad += actions[idx3][idx] * action_prob[idx3][idx] * performance[idx3] * states[idx3][idx2]
        grad_w1[-1].append(grad)

    grad_w1 = np.array(grad_w1)

    # Update RMS_grad
    self.RMS_grad['w1'] = 0.9 * self.RMS_grad['w1'] + 0.1 * grad_w1 * grad_w1

    # Finally, update weights with RMS_grad
    learning_rate = 0.001
    smoothing = 1e-8
    self.weights['w1'] = self.weights['w1'] - (learning_rate / np.sqrt(self.RMS_grad['w1'] + smoothing)) * grad_w1

    return loss


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
    self.hidden_layer_outputs = []
    self.RMS_grad = {
      'w1':np.zeros((10,4)),
      'b1':np.zeros((10,1)),
      'w2':np.zeros((1,10)),
      'b2':np.zeros((1,1))
      }


  def predict_cumulative_reward(self,history):
    cumulative_reward_history = []
    hidden_layer_output_history = []
    for step in history:
      hidden_layer_output = relu(np.matmul(self.weights['w1'],step['prev_state'])+self.weights['b1'])
      cumulative_reward = np.sum(np.matmul(self.weights['w2'],hidden_layer_output)+self.weights['b2'])
      hidden_layer_output_history.append(hidden_layer_output)
      cumulative_reward_history.append(cumulative_reward)
    self.hidden_layer_outputs = hidden_layer_output_history
    return cumulative_reward_history

  def update(self,predicted_cumulative_reward,performance,history):

    loss = 0.5 * sum(performance * performance)

    # Get states from history
    states = []
    for step in history: states.append(step['prev_state'])

    # Calculate gradients
    grad_w2 = []
    for idx in range(1):
      grad_w2.append([])
      for idx2 in range(10):
        grad = 0
        for idx3,step in enumerate(performance):
          grad += performance[idx3] * self.hidden_layer_outputs[idx3][idx2]
        grad_w2[-1].append(grad)
    grad_b2 = [[np.sum(performance)]]
    grad_w1 = []
    for idx in range(10):
      grad_w1.append([])
      for idx2 in range(4):
        if np.sum(np.sum(self.hidden_layer_outputs,axis=0),axis=0)[idx] > 0:
          grad = 0
          for idx3,step in enumerate(performance):
            grad += np.sum(performance[idx3] * self.weights['w2'] * states[idx3][idx2])
          grad_w1[-1].append(grad)
        else:
          grad_w1[-1].append(0)
    grad_b1 = []
    for idx in range(10):
      grad_b1.append([])
      if np.sum(np.sum(self.hidden_layer_outputs,axis=0),axis=0)[idx] > 0:
        grad_b1[-1].append(np.sum([np.sum(performance,axis=0)] * self.weights['w2']))
      else:
        grad_b1[-1].append(0)

    grad_w1 = np.array(grad_w1)
    grad_b1 = np.array(grad_b1)
    grad_w2 = np.array(grad_w2)
    grad_b2 = np.array(grad_b2)

    # Update RMS_grad
    self.RMS_grad['w1'] = 0.9 * self.RMS_grad['w1'] + 0.1 * grad_w1 * grad_w1
    self.RMS_grad['b1'] = 0.9 * self.RMS_grad['b1'] + 0.1 * grad_b1 * grad_b1
    self.RMS_grad['w2'] = 0.9 * self.RMS_grad['w2'] + 0.1 * grad_w2 * grad_w2
    self.RMS_grad['b2'] = 0.9 * self.RMS_grad['b2'] + 0.1 * grad_b2 * grad_b2

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

  policy = Policy()
  reward_model = RewardModel()

  # Run at most 10000 episodes for training
  for _ in range(10000):
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
    env.render()
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

  # Use performance and history to update policy
  policy_loss = policy.update(performance,history)

  print str(total_reward)+" ModelLoss: "+str(model_loss)+" PolicyLoss: "+str(policy_loss)

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
