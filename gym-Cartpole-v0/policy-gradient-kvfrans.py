"""
==================================================
kvfrans implementation of the policy gradient algorithm for Cartpole-v0
==================================================
## ## denotes my comments
## <> denotes tensors
## !! denotes unanswered question or comment
==================================================
"""
import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt

## This is simply the softmax function
## ie. normalize(exponential(x))
def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

## This defines the policy, as well as its optimizer
## !! Check out purpose of tf.variable_scope()
## !! Reason out which axis to use in expand_dims()
## <params> consist of weights combining states to give probability distribution of actions hence shape is [4,2] - 4 state-elements by 2 possible actions
## <state> stores the states
## <actions> stores the actions taken, consists of multiple 1-hot vectors
## <advantages> stores the "advantage" of each action, which is essentially the "profit" from that action
##   If intrinsic reward from that action is higher than average, advantage/"profit" is positive. If intrinsic reward is lower than average, advantage/"profit" is negative.
def policy_gradient():
    with tf.variable_scope("policy"):
        ##
        params = tf.get_variable("policy_parameters",[4,2])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,2])
        advantages = tf.placeholder("float",[None,1])
        ##
        ## Calculates probability distribution for actions as
        ## <probabilities> = softmax(<state> * <params>)
        linear = tf.matmul(state,params)
        probabilities = tf.nn.softmax(linear)
        ## good_probabilities might be a slight misnomer
        ## Here we calculate the loss function
        ## Essentially, we want to change the policy such that "correct" actions come up more frequently
        ## An action is more "correct" if its corresponding advantage is larger
        ##
        ## Suppose we are given state S

        ## Entering state S into the neural net gives an expected reward of 10.
        ## Entering state S into the policy gives a probability distribution of [0.75,0.25] for the actions.
        ##
        ## We stochastically chose action[0] and the actual future_reward is 15.
        ## The advantage is calculated as 15 - 10 = 5
        ##
        ## good_probabilities = 0.75
        ## eligibility = log(0.75) * 5 = -0.625
        ## loss = -reduce_sum(eligibility) = 0.625
        ##
        ## Suppose another policy gave probability distribution [0.25,0.75], given state S.
        ##
        ## And we stochastically chose action[1] and the actual future_reward is 8.
        ## The advantage is calculated as 8 - 10 = -2
        ##
        ## good_probabilities = 0.75
        ## eligibility = log(0.75) * -2 = 0.250
        ## loss = -reduce_sum(eligibility) = -0.250
        ##
        ## Notice that in the ideal case, where the policy and neural net converges,
        ## the advantage should be = 0
        ## This gives a loss of 0
        ## Hence the optimizer is trying to minimize the magnitude of the loss,
        ## adjusting the policy to increase the loss if it is negative
        ## and decrease the loss if it is positive
        ##
        good_probabilities = tf.reduce_sum(tf.multiply(probabilities, actions),reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probabilities, state, actions, advantages, optimizer

## This defines a neural net for predicting the expected intrinsic reward given a state
## It also defines an optimizer for improving the neural net
## The optimizer minimizes the difference between the predicted values and the "true-values"
## calculated from an episode (ie. the future_reward for each transition)
##
## This is used to generate a predicted reward,
## which is used as a baseline for judging the score of an action
## If the future_reward is higher than the predicted reward,
## the action is desirable, which is represented as
## advantage = future_reward - currentval
## The future_reward is than used to augment the neural net again for the next time-step
## <state> stores the states, used as input to the neural net
## <newvals> stores the future_reward values, used as 'true' labels
## The loss function is calculated as L2-loss(predicted_reward-actual_reward)
## L2-loss(x) = (reduce_sum(x^2))/2
def value_gradient():
    with tf.variable_scope("value"):
        ##
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        ## The neural net consists of a single 10-node hidden layer
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    ## Initialize policy and neural net
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []

    ## Runs an episode for at most 200 steps
    for _ in xrange(200):
        #env.render()
        # calculate policy
        ## Expands observation to fit <state> in policy_gradient()
        obs_vector = np.expand_dims(observation, axis=0)
        ## Calculates probability distribution from policy given the state
        probs = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        ## Stochastically chooses an action based on probability distribution
        action = 0 if random.uniform(0,1) < probs[0][0] else 1
        # record the transition
        ## Add observation to states
        states.append(observation)
        ## Add action taken (1-hot) to actions
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        ## Add transition (state, action, reward) to transitions
        transitions.append((old_observation, action, reward))
        totalreward += reward
        if done:
            print totalreward
            break
    ##
    ## Recall that transitions consist of (state, action, reward) elements
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        ## Essentially we want to calculate the 'future reward'
        ## Intuitively, a reward at step 100 is not only due to the action at step 100
        ## It is also due to cumulative effects of earlier steps
        ## This means that the action taken at step 10 should also be rewarded partially
        ## for the reward at step 100
        ## This portion calculates this as
        ## future_reward = current_reward + sum(reward_from_future_step * discount)
        ## where discount (<=1) decreases the further the future step
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in xrange(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.97
        ##
        ## As above, expands observation, this time to fit <state> in value_gradient()
        obs_vector = np.expand_dims(obs, axis=0)
        ## Calculates the predicted reward for a certain state
        ## Initially this is rubbish since the neural net is randomly initialized and untrained
        ## But over time, this will get closer to the 'true' value as the neural net improves
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        ## This calculates the score of an action
        ## by judging the future_reward of the action
        ## against the expected reward estimated by the neural net
        ## This 'expected reward' is based on the neural net as trained on past examples
        ## Hence, it represents a sort of average past performance
        ## or the expected reward based on past performance
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        ## Add future_reward to update_vals
        ## which is used to train the neural net for the next episode
        update_vals.append(future_reward)

    # update value function
    ## Update the neural net for the next episode
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    ## Update the policy for the next episode
    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward


env = gym.make('CartPole-v0')
##env.monitor.start('cartpole-hill/', force=True)
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
##rewards = []
for i in xrange(2000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    ##rewards.append(reward)
    if reward == 200:
        print "reward 200"
        print i
        break
t = 0
for _ in xrange(1000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    t += reward
print t / 1000
##env.monitor.close()
env.close()

##ax = plt.axes()
##ax.plot(range(len(rewards)),rewards)
##plt.show()
