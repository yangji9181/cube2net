import itertools
import pickle

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

	nx.draw_networkx_nodes(G, pos, nodelist=[node for node in nodes if node not in group], node_color='w', node_size=20)
	for g_id, color in colors:
		nx.draw_networkx_nodes(G, pos, nodelist=[node for node in nodes if node in group and group[node] == g_id],
		                       node_color=color, node_size=20)
	nx.draw_networkx_edges(G, pos, width=0.5)
	nx.draw_networkx_labels(G, pos, font_size=6)
	plt.savefig(cwd + suffix + '3.png', dpi=400)



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


class Graph(object):
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

	def colored(self):
		return set(self.neighbors_baseline.keys())

	def graph1(self):
		edges = []
		for node in self.connected:
			neighbors = self.neighbors_rl[node] & self.connected
			if bool(neighbors):
				for neighbor in neighbors:
					edges.append((node, neighbor))
		return list(self.connected), edges

	def graph2(self):
		nodes, edges = set(), set()
		colored = self.colored()
		for node, neighbors in self.neighbors_rl.items():
			if node not in self.neighbors_baseline and bool(neighbors & colored):
				nodes.add(node)
				for neighbor in neighbors:
					if neighbor in self.neighbors_baseline:
						nodes.add(neighbor)
		for node, neighbors in self.neighbors_rl.items():
			if node in nodes and node in self.neighbors_baseline:
				intersection = neighbors & nodes
				for neighbor in intersection:
					edges.add((node, neighbor))

		baseline_nodes = nodes & colored
		baseline_edges = set()
		for node, neighbors in self.neighbors_baseline.items():
			if node in baseline_nodes:
				intersection = neighbors & baseline_nodes
				for neighbor in intersection:
					baseline_edges.add((node, neighbor))

		return list(nodes), list(edges), list(baseline_nodes), list(baseline_edges)

	def one(self):
		nodes = set()
		baseline_nodes = set()
		colored = self.colored()
		for node, neighbors in self.neighbors_rl.items():
			if node not in self.neighbors_baseline:
				intersect = neighbors & colored
				if len(intersect) > 1:
					should_add = False
					for pair in itertools.combinations(intersect, 2):
						if pair[1] not in self.neighbors_baseline[pair[0]] \
								and self.author_labels[pair[0]] != self.author_labels[pair[1]]:
							should_add = True
							nodes.add(pair[0])
							nodes.add(pair[1])
							baseline_nodes.add(pair[0])
							baseline_nodes.add(pair[1])
					if should_add:
						nodes.add(node)
		nodes = set(list(nodes)[:60])
		baseline_nodes = set(list(baseline_nodes)[:40])
		edges = self.edges(nodes)
		baseline_edges = self.edges(baseline_nodes)
		return list(nodes), list(edges), list(baseline_nodes), list(baseline_edges)

	def edges(self, nodes):
		edges = set()
		for node, neighbors in self.neighbors_rl.items():
			if node in nodes:
				intersect = neighbors & nodes
				if bool(intersect):
					for neighbor in intersect:
						if node in self.author_labels or neighbor in self.author_labels:
							edges.add((node, neighbor))
		return edges


if __name__ == '__main__':
	cwd = 'data/'
	graph = Graph()
	authors, links, baseline_authors, baseline_links = graph.one()
	plot(baseline_authors, baseline_links, graph.author_labels, 'baseline')
	plot(authors, links, graph.author_labels, 'rl')
