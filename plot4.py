import itertools
import pickle
import random
import matplotlib
import networkx as nx

matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from config import *


def plot(nodes, edges, group, suffix):
	colors = [(0, 'w'), (1, 'r'), (2, 'g'), (3, 'b'), (4, 'y')]
	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	pos = nx.spring_layout(G)

	nx.draw_networkx_nodes(G, pos, nodelist=[node for node in nodes if node not in group], node_color='w', node_size=10)
	for g_id, color in colors:
		nx.draw_networkx_nodes(G, pos, nodelist=[node for node in nodes if node in group and group[node] == g_id],
		                       node_color=color, node_size=10)
	nx.draw_networkx_edges(G, pos, width=0.2)
	nx.draw_networkx_labels(G, pos, font_size=5)
	plt.savefig(cwd + suffix + '.png', dpi=1000)



def read_graph(suffix):
	with open(cwd + 'nodes_' + suffix + '.pkl', 'rb') as f:
		nodes = pickle.load(f)
	with open(cwd + 'edges_' + suffix + '.pkl', 'rb') as f:
		edges = pickle.load(f)
	return nodes, edges


def read_test():
	test_authors = defaultdict(set)
	with open(args.test_file) as f:
		for line in f:
			splits = line.rstrip().split('\t')
			test_authors[splits[0].replace('_', ' ')] = int(splits[1])
	return test_authors


class Network(object):
	def __init__(self):
		nodes_baseline, edges_baseline = read_graph('baseline')
		nodes_rl, edges_rl = read_graph('rl')
		self.author_labels = read_test()

		self.neighbors_baseline = {node: set() for node in nodes_baseline}
		self.neighbors_rl = {node: set() for node in nodes_rl}

		for edge in edges_baseline:
			self.neighbors_baseline[edge[0]].add(edge[1])
			self.neighbors_baseline[edge[1]].add(edge[0])

		for edge in edges_rl:
			self.neighbors_rl[edge[0]].add(edge[1])
			self.neighbors_rl[edge[1]].add(edge[0])

		self.dangling = set([node for node in self.neighbors_baseline if not bool(self.neighbors_baseline[node])])
		self.connected = set([node for node in self.neighbors_rl if bool(self.neighbors_rl[node] & self.dangling)]) | self.dangling


	def edges(self, nodes):
		edges = set()
		for node, neighbors in self.neighbors_rl.items():
			if node in nodes:
				intersect = neighbors & nodes
				if bool(intersect):
					for neighbor in intersect:
						self.add_edge(edges, node, neighbor)
		return edges

	def colored(self):
		return set(self.neighbors_baseline.keys())

	def is_connected(self, n1, n2, order=0):
		if order == 0:
			return n1 == n2
		if len(self.neighbors_baseline[n1]) == 0:
			return False
		return reduce(lambda x, y: x or y, [self.is_connected(n, n2, order - 1) for n in self.neighbors_baseline[n1]])

	def baseline(self):
		nodes = set()
		colored = self.colored()
		for node, nbs in self.neighbors_rl.items():
			neighbors = nbs & colored
			if node not in self.neighbors_baseline:
				for pair in itertools.combinations(neighbors, 2):
					if self.author_labels[pair[0]] == self.author_labels[pair[1]]:
						if not self.is_connected(pair[0], pair[1], 4):
							nodes.add(pair[0])
							nodes.add(pair[1])
			else:
				for neighbor in neighbors:
					if self.author_labels[node] != self.author_labels[neighbor]:
						nodes.add(node)
						nodes.add(neighbor)
		edges = self.edges(nodes)
		return list(nodes), list(edges)

	def add_edge(self, edges, n1, n2):
		if (n1, n2) not in edges and (n2, n1) not in edges:
			edges.add((n1, n2))

	def rl1(self, baseline_nodes, baseline_edges):
		colored = self.colored()
		nodes = set(baseline_nodes)
		edges = set(baseline_edges)
		for (n1, n2) in itertools.combinations(nodes, 2):
			if self.author_labels[n1] == self.author_labels[n2] and n2 not in self.neighbors_baseline[n1]:
				intersect = self.neighbors_rl[n1] & self.neighbors_rl[n2]
				for nb in intersect:
					if nb not in colored:
						nodes.add(nb)
						self.add_edge(edges, n1, nb)
						self.add_edge(edges, n2, nb)
		return list(nodes), list(edges)

	def rl2(self, baseline_nodes, baseline_edges):
		colored = self.colored()
		nodes = set(baseline_nodes)
		edges = set(baseline_edges)
		for edge in baseline_edges:
			if self.author_labels[edge[0]] != self.author_labels[edge[1]]:
				for nb1 in self.neighbors_rl[edge[0]]:
					if nb1 not in colored:
						if random.uniform(0, 1) > 0.8:
							nodes.add(nb1)
							self.add_edge(edges, edge[0], nb1)
				for nb2 in self.neighbors_rl[edge[1]]:
					if nb2 not in colored:
						if random.uniform(0, 1) > 0.8:
							nodes.add(nb2)
							self.add_edge(edges, edge[1], nb2)
		return list(nodes), list(edges)

	def rl3(self, baseline_nodes, baseline_edges):
		return None, None


if __name__ == '__main__':
	cwd = 'data/'
	graph = Network()
	baseline_authors, baseline_links = graph.baseline()
	# plot(baseline_authors, baseline_links, graph.author_labels, 'baseline')
	# authors1, links1 = graph.rl1(baseline_authors, baseline_links)
	# plot(authors1, links1, graph.author_labels, 'rl1')
	authors2, links2 = graph.rl2(baseline_authors, baseline_links)
	plot(authors2, links2, graph.author_labels, 'rl2')
