import numpy as np
from NN import *
from tqdm import tqdm


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

		self.build_train(tf.nn.embedding_lookup(self.cell_embed, self.action), self.reward_to_go, value, policy, policy_old)
		self.decision = self.build_plan(policy)

	def build_train(self, action, reward_to_go, value, policy_mean, policy_mean_old):
		advantage = reward_to_go - tf.stop_gradient(value)
		# Gaussian policy with identity matrix as covariance mastrix
		ratio = tf.exp(0.5 * tf.reduce_sum(tf.squared_difference(action, policy_mean_old), axis=-1) -
		               0.5 * tf.reduce_sum(tf.squared_difference(action, policy_mean), axis=-1))
		surr_loss = tf.minimum(ratio * advantage, tf.clip_by_value(ratio, 1.0 - self.params.clip_epsilon, 1.0 + self.params.clip_epsilon) * advantage)
		surr_loss = -tf.reduce_sum(surr_loss, axis=-1)
		v_loss = tf.reduce_sum(tf.squared_difference(reward_to_go, value), axis=-1)

		critic_optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
		actor_optimizer = tf.train.AdamOptimizer(self.params.learning_rate)
		self.critic_step = critic_optimizer.minimize(v_loss)
		self.actor_step = actor_optimizer.minimize(surr_loss)

	def build_plan(self, policy_mean):
		policy = tf.distributions.Normal(policy_mean, tf.ones([self.params.embed_dim], tf.float32))
		action_embed = policy.sample()
		return tf.argmin(tf.reduce_sum(
			tf.squared_difference(tf.expand_dims(action_embed, axis=1), tf.expand_dims(self.cell_embed, axis=0)), axis=-1), axis=-1)

	def value_policy(self, state):
		hidden = state
		for i, dim in enumerate(self.params.hidden_dim):
			hidden = fully_connected(hidden, dim, 'policy_value_' + str(i))
		return hidden

	def value(self, hidden):
		return fully_connected(hidden, 1, 'policy_value_o')

	def policy(self, hidden):
		return fully_connected(hidden, self.params.embed_dim, 'policy_o')

	# the number of trajectories sampled is equal to batch size
	def collect_trajectory(self, sess):
		ret_states, ret_actions = [], []
		batch_states = [self.environment.initial_state() for _ in range(self.params.batch_size)]
		feed_state = np.array([self.environment.state_embed(list(s)) for s in batch_states])
		batch_actions = []
		for i in range(self.params.trajectory_length):
			ret_states.append(feed_state)
			action = sess.run(self.decision, feed_dict={self.state: feed_state})
			ret_actions.append(action)
			batch_actions.append(list(action))
			for i, state in enumerate(batch_states):
				state.add(action[i])
				feed_state[i] = self.environment.state_embed(list(state))
		ret_rewards = [self.environment.trajectory_reward(s, a) for s, a in zip(batch_states, batch_actions)]
		return zip(ret_states, ret_actions, ret_rewards)

	def train(self, sess):
		sess.run(tf.global_variables_initializer())
		for _ in tqdm(range(self.params.epoch), ncols=100):
			trajectories = self.collect_trajectory(sess)
			for _ in tqdm(range(self.params.critic_step), ncols=100):
				for state, action, reward_to_go in trajectories:
					sess.run(self.critic_step, feed_dict={self.state: state, self.reward_to_go: reward_to_go})
			for _ in tqdm(range(self.params.actor_step), ncols=100):
				for state, action, reward_to_go in trajectories:
					sess.run(self.actor_step, feed_dict={self.state: state, self.action: action, self.reward_to_go : reward_to_go})
			sess.run(self.assign_ops)

	def plan(self, sess):
		batch_states = [self.environment.initial_state() for _ in range(self.params.batch_size)]
		feed_state = np.array([self.environment.state_embed(list(s)) for s in batch_states])
		for _ in range(self.params.trajectory_length):
			action = sess.run(self.decision, feed_dict={self.state: feed_state})
			for i, state in enumerate(batch_states):
				state.add(action[i])
				feed_state[i] = self.environment.state_embed(list(state))
		return np.average(np.array([self.environment.total_reward(s) for s in batch_states]))
