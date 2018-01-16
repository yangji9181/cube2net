import numpy as np
from NN import *
from tqdm import tqdm

class Agent(object):
	def __init__(self, params, environment):
		self.params = params
		self.environment = environment
		self.build()

	def build(self):
		self.state_embed = tf.placeholder(tf.float32, [None, 2 * self.params.embed_dim])

		self.weights, self.weights_t = {}, {}
		hidden = self.state_embed
		hidden_t = self.state_embed
		for i, dim in enumerate(self.params.hidden_dim):
			hidden, W, b = fully_connected(hidden, dim, 'prediction_' + str(i + 1))
			hidden_t, W_t, b_t = fully_connected(hidden_t, dim, 'target_' + str(i + 1))
			self.weights['W_' + str(i + 1)], self.weights['W_' + str(i + 1)] = W, b
			self.weights_t['W_' + str(i + 1)], self.weights_t['W_' + str(i + 1)] = W_t, b_t
		# use sigmoid when reward is clustering coefficient
		q, W, b = fully_connected(hidden, 1, 'prediction_q', activation='sigmoid')
		q_t, W_t, b_t = fully_connected(hidden_t, 1, 'target_q', activation='sigmoid')
		self.weights['W_o'], self.weights['b_o'] = W, b
		self.weights_t['W_o'], self.weights_t['b_o'] = W_t, b_t

		self.assign = [tf.assign(self.weights_t[name], self.weights[name]) for name in self.weights]

		self.q_t = q_t
		self.q_v = tf.placeholder(tf.float32, [None])
		loss = tf.reduce_mean(tf.squared_difference(self.q_t, self.q_v))
		optimizer = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate)
		self.step = optimizer.minimize(loss)

	def update_target(self, sess):
		for op in self.assign:
			sess.run(op)

	def train(self):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for _ in tqdm(range(self.params.epoch), ncols=100):
				self.update_target(sess)
				# completely off policy
				for _ in tqdm(range(self.params.k_step), ncols=100):
					feed_state, feed_q = [], []
					for sample in self.environment.sample():
						s, a, n, r = sample
						feed_state.append(self.environment.state_embed(s))
						nns = self.environment.next_states(n)
						state_embeds = []
						for nn in nns:
							state_embeds.append(self.environment.state_embed(nn))
						state_embeds = np.array(state_embeds)
						feed_q.append(r + np.max(
							sess.run(self.q_t, feed_dict={self.state_embed: np.array(state_embeds)})))
					sess.run(self.step, feed_dict={self.state_embed: np.array(feed_state), self.q_v: feed_q})

	def play(self):
		pass
