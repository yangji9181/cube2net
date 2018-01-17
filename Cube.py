import itertools
import pickle
import networkx as nx


class Cube(object):
	# state is a set of cells, action is a single cell
	def reward(self, state, action):
		G1, G2 = nx.Graph(), nx.Graph()
		for v, y in state:
			papers = self.cell_venue[v] & self.cell_year[y]
			for paper in papers:
				for pair in itertools.combinations(self.paper_author[paper], 2):
					G1.add_edge(pair[0], pair[1])
					G2.add_edge(pair[0], pair[1])

		v, y = action
		papers = self.cell_venue[v] & self.cell_year[y]
		for paper in papers:
			for pair in itertools.combinations(self.paper_author[paper], 2):
				G2.add_edge(pair[0], pair[1])

		return nx.average_clustering(G2) - nx.average_clustering(G1)

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
