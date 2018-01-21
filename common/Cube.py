import itertools
import pickle
import networkx as nx


class Cube(object):
	# state is a set of cells, action is a single cell
	def reward(self, state, action):
		G1, G2 = nx.Graph(), nx.Graph()
		for cell in state:
			self.add_cell(G1, cell)
			self.add_cell(G2, cell)
		self.add_cell(G2, action)
		return nx.average_clustering(G2) - nx.average_clustering(G1)

	# compute reward to go
	def trajectory_reward(self, state, actions):
		G = nx.Graph()
		for cell in state:
			self.add_cell(G, cell)
		rewards = [nx.average_clustering(G)]

		for cell in actions:
			self.add_cell(G, cell)
			rewards.append(nx.average_clustering(G))
		total = rewards[-1]
		rewards = [total - r for r in rewards]
		return rewards[:-1]

	# mutate G
	def add_cell(self, G, cell):
		v, y = cell
		papers = self.cell_venue[v] & self.cell_year[y]
		for paper in papers:
			for pair in itertools.combinations(self.paper_author[paper], 2):
				G.add_edge(pair[0], pair[1])

	def total_reward(self, state):
		G = nx.Graph()
		for v, y in state:
			papers = self.cell_venue[v] & self.cell_year[y]
			for paper in papers:
				for pair in itertools.combinations(self.paper_author[paper], 2):
					G.add_edge(pair[0], pair[1])
		return nx.average_clustering(G)


	@staticmethod
	def load_cube(path):
		with open(path, 'rb') as f:
			return pickle.load(f)
