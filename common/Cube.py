import itertools
import pickle
import networkx as nx


class Cube(object):
	def initial_state(self, path, threshold):
		authors = set()
		with open(path) as f:
			for line in f:
				authors.add(line.rstrip().split('\t')[0])
		ids = []
		for id, cell_authors in self.id_to_author.items():
			if len(cell_authors & authors) > threshold:
				ids.append(id)
		return set(ids)

	# state is a set of cells, action is a single cell
	def reward(self, state, action, func):
		G1, G2 = nx.Graph(), nx.Graph()
		for cell in state:
			self.add_cell(G1, cell)
			self.add_cell(G2, cell)
		self.add_cell(G2, action)
		return getattr(nx, func)(G2) - getattr(nx, func)(G1)

	# compute reward to go
	def trajectory_reward(self, state, actions, func):
		G = nx.Graph()
		for cell in state:
			self.add_cell(G, cell)
		rewards = [getattr(nx, func)(G)]

		for cell in actions:
			self.add_cell(G, cell)
			rewards.append(getattr(nx, func)(G))
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

	def total_reward(self, state, func):
		G = nx.Graph()
		for v, y in state:
			papers = self.cell_venue[v] & self.cell_year[y]
			for paper in papers:
				for pair in itertools.combinations(self.paper_author[paper], 2):
					G.add_edge(pair[0], pair[1])
		return getattr(nx, func)(G)


	@staticmethod
	def load_cube(path):
		with open(path, 'rb') as f:
			return pickle.load(f)
