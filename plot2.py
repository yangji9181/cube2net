import pickle
import time
import os
import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from cube.utils import DblpEval
from config import *
from Environment import *
from PPO import *
from cube.cube_construction import DblpCube
from Cube import Cube



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
	nx.draw_networkx_labels(G, pos, font_size=8)
	plt.savefig(cwd + suffix + '2.png')


def dump_graph(nodes, edges, suffix):
	with open(cwd + 'nodes_' + suffix + '2.pkl', 'wb') as f:
		pickle.dump(nodes, f)
	with open(cwd + 'edges_' + suffix + '2.pkl', 'w') as f:
		pickle.dump(edges, f)


def read_graph(suffix):
	with open(cwd + 'nodes_' + suffix + '2.pkl', 'rb') as f:
		nodes = pickle.load(f)
	with open(cwd + 'edges_' + suffix + '2.pkl', 'rb') as f:
		edges = pickle.load(f)
	return nodes, edges


def parse_links(links):
	results = []
	for link in links:
		results.append(tuple(link.split(',')))
	return results


if __name__ == '__main__':
	cwd = 'data/'
	cube = None

	test_authors = defaultdict(set)
	with open(args.test_file) as f:
		for line in f:
			splits = line.rstrip().split('\t')
			test_authors[splits[0].replace('_', ' ')] = int(splits[1])

	if os.path.isfile(cwd + 'nodes_baseline2.pkl') and os.path.isfile(cwd + 'edges_baseline2.pkl') \
			and os.path.isfile(cwd + 'nodes_rl2.pkl') and os.path.isfile(cwd + 'edges_rl2.pkl'):
		intersect_authors, intersect_links = read_graph('baseline')
		coauthors, colinks = read_graph('rl')
	else:
		with open('cube/models/step3.pkl', 'r') as f:
			cube = pickle.load(f)

		environment = Environment(args)
		tf.reset_default_graph()
		os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device_id)
		with tf.device('/gpu:0'):
			agent = PPO(args, environment)
		with tf.Session(config=tf.ConfigProto(
				allow_soft_placement=True,
				gpu_options=tf.GPUOptions(
					per_process_gpu_memory_fraction=0.5,
					allow_growth=True))) as sess:
			agent.train(sess)
			authors, reward, actions = agent.plan(sess, union=False)
		intersect_authors = authors & cube.author0
		intersect_links = parse_links(DblpEval.author_links(cube, intersect_authors))
		coauthors = authors & DblpEval.coauthors(cube, intersect_authors, order=1)
		colinks = parse_links(DblpEval.author_links(cube, coauthors))
		dump_graph(intersect_authors, intersect_links, 'baseline')
		dump_graph(coauthors, colinks, 'rl')

	plot(intersect_authors, intersect_links, test_authors, 'baseline')
	plot(coauthors, colinks, test_authors, 'rl')

