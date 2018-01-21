import numpy as np
from NN import *
from tqdm import tqdm

class DQN(object):
	def __init__(self, params, environment):
		self.params = params
		self.environment = environment
		self.build()

	def build(self):
		self.state = tf.placeholder(tf.float32, [None, self.params.embed_dim])
		self.action = tf.placeholder(tf.int32, [None])
		self.next = tf.placeholder(tf.float32, [None, self.params.embed_dim])
		self.reward = tf.placeholder(tf.float32, [None])

		weights, weights_t = {}, {}

		with tf.variable_scope('prediction'):
			hidden = self.state
			for i, dim in enumerate(self.params.hidden_dim):
				hidden, W, b = fully_connected(hidden, dim, 'prediction_' + str(i + 1))
				weights['W_' + str(i + 1)], weights['W_' + str(i + 1)] = W, b
			# use sigmoid when reward is clustering coefficient
			Q, W, b = fully_connected(hidden, self.environment.action_size, 'prediction_q', activation='sigmoid')
			weights['W_o'], weights['b_o'] = W, b

		with tf.variable_scope('target'):
			hidden_t = self.next
			for i, dim in enumerate(self.params.hidden_dim):
				hidden_t, W_t, b_t = fully_connected(hidden_t, dim, 'target_' + str(i + 1))
				weights_t['W_' + str(i + 1)], weights_t['W_' + str(i + 1)] = W_t, b_t
			# use sigmoid when reward is clustering coefficient
			Q_t, W_t, b_t = fully_connected(hidden_t, self.environment.action_size, 'target_q', activation='sigmoid')
			weights_t['W_o'], weights_t['b_o'] = W_t, b_t

		self.assign_ops = tf.group(*[tf.assign(weights_t[name], weights[name]) for name in weights])

		self.Q = Q
		action_one_hot = tf.one_hot(self.action, self.environment.action_size, 1.0, 0.0)
		self.loss = tf.reduce_mean(tf.squared_difference(tf.reduce_sum(Q * action_one_hot, axis=1),
		                                            self.reward +  tf.reduce_max(Q_t, axis=1)))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
		self.step = optimizer.minimize(
			self.loss, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='prediction'))

		# self.print_variable()

	def print_variable(self):
		for var in tf.trainable_variables():
			print(var.name, var.get_shape())

	def train(self, sess):
		sess.run(tf.global_variables_initializer())
		for _ in tqdm(range(self.params.epoch), ncols=100):
			sess.run(self.assign_ops)
			# completely off policy
			for _ in tqdm(range(self.params.k_step), ncols=100):
				feed_state, feed_action, feed_next, feed_reward = [], [], [], []
				for sample in self.environment.sample():
					s, a, n, r = sample
					feed_state.append(self.environment.state_embed(s))
					feed_action.append(a)
					feed_next.append(self.environment.state_embed(n))
					feed_reward.append(r)
				sess.run(self.step, feed_dict={self.state: np.array(feed_state),
				                               self.action: np.array(feed_action),
				                               self.next: np.array(feed_next),
				                               self.reward: np.array(feed_reward)})

	def plan(self, sess):
		state = self.environment.initial_state()
		for _ in range(self.params.trajectory_length):
			feed_state = np.expand_dims(self.environment.state_embed(list(state)), axis=0)
			actions = np.argsort(-sess.run(self.Q, feed_dict={self.state: feed_state})[0])
			for a in actions:
				if a not in state:
					action = int(a)
					break
			state.add(action)
		return self.environment.total_reward(state)
