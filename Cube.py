import pickle
import networkx as nx
import numpy as np
from copy import deepcopy


class Cube(object):
	def initial_state(self, path, low_limit, high_limit, debug=False):
		if debug:
			return set(list(np.random.choice(len(self.id_to_cell), 10, replace=False)))
		test_authors = set()
		with open(path) as f:
			for line in f:
				test_authors.add(line.rstrip().split('\t')[0].replace('_', ' '))

		# self.author_1st = deepcopy(test_authors)
		# for links in self.id_to_link:
		# 	for pair in links:
		# 		if pair[0] in self.author_1st:
		# 			self.author_1st.add(pair[1])
		# 		elif pair[1] in self.author_1st:
		# 			self.author_1st.add(pair[0])

		self.init_authors = test_authors
		ids = []
		for id, _ in enumerate(self.id_to_cell):
			count = len(self.id_to_author[id] & test_authors)
			if count >= low_limit and count < high_limit:
				ids.append(id)
		return set(ids)

	def all_authors(self, state):
		authors = deepcopy(self.init_authors)
		for id in state:
			authors |= self.id_to_author[id]
		return authors

	# compute reward to go
	def trajectory_reward(self, state, actions, params):
		G = nx.Graph()
		for cell in state:
			self.add_cell(G, cell)
		rewards = [self.reward(G, params)]

		for cell in actions:
			self.add_cell(G, cell)
			rewards.append(self.reward(G, params))
		total = rewards[-1]
		rewards = [total - r for r in rewards]
		return rewards[:-1]

	# mutate G
	def add_cell(self, G, cell):
		for pair in self.id_to_link[cell]:
			G.add_edge(pair[0], pair[1])

	def total_reward(self, state, params):
		G = nx.Graph()
		for cell in state:
			self.add_cell(G, cell)
		return self.reward(G, params)

	def reward(self, G, params):
		nodes = set(nx.nodes(G))
		return params.transitivity_c * nx.transitivity(G) + \
		       params.connectivity_c * float(len(nodes & self.init_authors)) / float(len(nodes))


	@staticmethod
	def load_cube(path):
		with open(path, 'rb') as f:
			return pickle.load(f)
