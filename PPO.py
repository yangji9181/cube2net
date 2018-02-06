import numpy as np
from copy import deepcopy
from NN import *
from tqdm import tqdm
from random import shuffle


class PPO(object):
	def __init__(self, params, environment):
		self.params = params
		self.environment = environment
		self.build()

	def build(self):
		self.cell_embed = tf.Variable(self.environment.cell_embed, trainable=False, dtype=tf.float32)
		self.state = tf.placeholder(tf.float32, [None, self.params.embed_dim])
		self.action = tf.placeholder(tf.int32, [None])
		self.reward_to_go = tf.placeholder(tf.float32, [None])

		hidden = self.value_policy(self.state)
		value = self.value(hidden)
		with tf.variable_scope('new'):
			policy = self.policy(hidden)
		with tf.variable_scope('old'):
			policy_old = self.policy(hidden)
		assign_ops = []
		for new, old in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='new'),
		                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='old')):
			assign_ops.append(tf.assign(old, new))
		self.assign_ops = tf.group(*assign_ops)

		# use scaled std of embedding vectors as policy std
		sigma = tf.Variable(self.environment.sigma * self.params.sigma_ratio, trainable=False, dtype=tf.float32)
		print(self.environment.sigma * self.params.sigma_ratio)
		self.build_train(tf.nn.embedding_lookup(self.cell_embed, self.action), self.reward_to_go, value, policy, policy_old, sigma)
		self.decision = self.build_plan(policy, sigma)

	def build_train(self, action, reward_to_go, value, policy_mean, policy_mean_old, sigma):
		advantage = reward_to_go - tf.stop_gradient(value)
		# Gaussian policy with identity matrix as covariance mastrix
		ratio = tf.exp(0.5 * tf.reduce_sum(tf.square((action - policy_mean_old) * sigma), axis=-1) -
		               0.5 * tf.reduce_sum(tf.square((action - policy_mean) * sigma), axis=-1))
		surr_loss = tf.minimum(ratio * advantage, tf.clip_by_value(ratio, 1.0 - self.params.clip_epsilon, 1.0 + self.params.clip_epsilon) * advantage)
		surr_loss = -tf.reduce_mean(surr_loss, axis=-1)
		v_loss = tf.reduce_mean(tf.squared_difference(reward_to_go, value), axis=-1)

		optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
		self.step = optimizer.minimize(surr_loss + self.params.c_value * v_loss)

	def build_plan(self, policy_mean, sigma):
		policy = tf.distributions.Normal(policy_mean, sigma)
		policy = tf.Print(policy, [policy])
		action_embed = policy.sample()
		return tf.argmin(tf.reduce_sum(
			tf.squared_difference(tf.expand_dims(action_embed, axis=1), tf.expand_dims(self.cell_embed, axis=0)), axis=-1), axis=-1)

	def value_policy(self, state):
		hidden = state
		for i, dim in enumerate(self.params.hidden_dim):
			hidden = fully_connected(hidden, dim, 'policy_value_' + str(i))
		return hidden

	def value(self, hidden):
		return fully_connected(hidden, 1, 'policy_value_o', activation='linear')

	def policy(self, hidden):
		return fully_connected(hidden, self.params.embed_dim, 'policy_o')

	# the number of trajectories sampled is equal to batch size
	def collect_trajectory(self, sess):
		ret_states = []
		initial_states = [self.environment.init_state] * self.params.batch_size
		batch_states = [deepcopy(state) for state in initial_states]
		feed_state = np.array([self.environment.state_embed(list(s)) for s in batch_states])
		batch_actions = []
		for i in range(self.params.trajectory_length):
			ret_states.append(feed_state)
			action = sess.run(self.decision, feed_dict={self.state: feed_state})
			batch_actions.append(action)
			for i, state in enumerate(batch_states):
				state.add(action[i])
				feed_state[i] = self.environment.state_embed(list(state))
		ret_states = list(np.transpose(np.array(ret_states), (1, 0, 2)))
		batch_actions = list(np.transpose(batch_actions))
		return self.environment.reward_multiprocessing(ret_states, initial_states, batch_actions)

	def train(self, sess):
		sess.run(tf.global_variables_initializer())
		for _ in tqdm(range(self.params.epoch), ncols=100):
			states, actions, rewards = self.collect_trajectory(sess)
			indices = range(self.params.trajectory_length * self.params.batch_size)
			shuffle(indices)
			batch_size = self.params.trajectory_length * self.params.batch_size / self.params.step
			for _ in tqdm(range(self.params.outer_step), ncols=100):
				for i in range(self.params.step):
					batch_indices = indices[i * batch_size : (i + 1) * batch_size]
					sess.run(self.step,
					         feed_dict={self.state: states[batch_indices], self.action: actions[batch_indices], self.reward_to_go: rewards[batch_indices]})
			sess.run(self.assign_ops)

	def plan(self, sess):
		state = self.environment.init_state
		actions = []
		for _ in range(self.params.trajectory_length):
			feed_state = np.expand_dims(self.environment.state_embed(list(state)), axis=0)
			action = sess.run(self.decision, feed_dict={self.state: feed_state})
			actions.append(action[0])
			state.add(action[0])
		return self.environment.convert_state(state), self.environment.total_reward(state), actions
