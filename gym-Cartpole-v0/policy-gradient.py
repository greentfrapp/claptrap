import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt

def main():
  env = gym.make('CartPole-v0')
  #log_directory = '/tmp/cartpole-hill-climbing'
  #env = wrappers.Monitor(env,log_directory,force=True)

  total_reward = 0

  # Learning parameters
  episode_no = 0
  batch_size = 10
  decay_rate = 0.9
  learning_rate = 1e-3
  running_avg_reward = 0
  batch_reward = []

  # Neural network parameters
  num_hidden_layer_neurons = 10
  input_dimensions = 4
  weights = {
    '1': np.random.randn(num_hidden_layer_neurons,input_dimensions)/np.sqrt(input_dimensions),
    '2': np.random.randn(num_hidden_layer_neurons)/np.sqrt(num_hidden_layer_neurons)
  }

  # Plot axes
  episodes = []
  rewards = []

  # START - TOREAD
  expectation_g_squared = {}
  g_dict = {}
  for layer_name in weights.keys():
    expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
    g_dict[layer_name] = np.zeros_like(weights[layer_name])
  # END - TOREAD

  episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

  observation = env.reset()

  while episode_no<=10000:

    env.render()

    hidden_layer_values,action_probability = boomboom(observation, weights)

    episode_observations.append(observation)
    episode_hidden_layer_values.append(hidden_layer_values)

    action = choose_action(action_probability)

    observation,reward,done,info = env.step(action)

    total_reward += reward
    episode_rewards.append(total_reward)

    fake_label = 1 if action == 1 else 0
    loss_function_gradient = fake_label - action_probability
    episode_gradient_log_ps.append(loss_function_gradient)

    if done:

      episode_no += 1

      episode_hidden_layer_values = np.stack(episode_hidden_layer_values,0)
      episode_observations = np.stack(episode_observations,0)
      episode_gradient_log_ps = np.stack(episode_gradient_log_ps,0)
      episode_rewards = np.stack(episode_rewards,0)

      episode_gradient_log_ps = discount_with_rewards(episode_gradient_log_ps,episode_rewards,running_avg_reward)

      gradient = get_gradient(episode_gradient_log_ps,episode_hidden_layer_values,episode_observations,weights)

      for layer_name in gradient: g_dict[layer_name] += gradient[layer_name]
      if episode_no % batch_size == 0:
        update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)
        running_avg_reward = np.mean(batch_reward)

      #print weights['1'][0]
      print total_reward

      episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] # reset values

      episodes.append(episode_no)
      rewards.append(total_reward)
      batch_reward.append(total_reward)
      total_reward = 0
      observation = env.reset()

  env.close()

  ax = plt.axes()
  ax.plot(episodes,rewards)
  plt.show()

def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))

def relu(vector):
  vector[vector < 0] = 0
  return vector

def boomboom(observation,weights):
  hidden_layer_values = np.dot(weights['1'], observation)
  hidden_layer_values = relu(hidden_layer_values)
  output_layer_values = np.dot(hidden_layer_values, weights['2'])
  output_layer_values = sigmoid(output_layer_values)
  return hidden_layer_values,output_layer_values

def choose_action(probability):
  random_value = np.random.uniform()
  if random_value < probability:
    return 0
  else:
    return 1

def discount_with_rewards(gradient_log_p, episode_rewards, threshold):
  """ discount the gradient with the normalized rewards """
  discounted_episode_rewards = episode_rewards
  # standardize the rewards to be unit normal (helps control the gradient estimator variance)
  #if discounted_episode_rewards[-1]<threshold:
  #  discounted_episode_rewards = -discounted_episode_rewards
  #discounted_episode_rewards -= np.mean(discounted_episode_rewards)
  #discounted_episode_rewards /= np.std(discounted_episode_rewards)
  #return gradient_log_p * discounted_episode_rewards
  if discounted_episode_rewards[-1]<threshold: return -gradient_log_p * (threshold - discounted_episode_rewards[-1])
  else: return gradient_log_p * (discounted_episode_rewards[-1])

def get_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
  # START - TOREAD
  delta_L = gradient_log_p
  dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
  delta_l2 = np.outer(delta_L, weights['2'])
  delta_l2 = relu(delta_l2)
  dC_dw1 = np.dot(delta_l2.T, observation_values)
  # END - TOREAD
  return {
      '1': dC_dw1,
      '2': dC_dw2
  }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
  #print g_dict
  #print weights
  # START - TOREAD
  epsilon = 1e-5
  for layer_name in weights.keys():
    g = g_dict[layer_name]
    expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
    weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
    g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer
   # END - TOREAD

if __name__=="__main__": main()
