import pickle
import time
import os
import networkx as nx
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cube.utils import DblpEval
from config import *
from Environment import *
from PPO import *
from cube.cube_construction import DblpCube
from Cube import Cube



def plot(nodes, edges, suffix):
	G = nx.Graph()
	G.add_nodes_from(nodes)
	G.add_edges_from(edges)
	nx.draw(G)
	plt.savefig(cwd + 'suffix.png')


def dump_graph(nodes, edges, suffix):
	with open(cwd + 'nodes_' + suffix + '.txt', 'w') as f:
		for node in nodes:
			f.write(node + '\n')
	with open(cwd + 'edges_' + suffix + '.txt', 'w') as f:
		for edge in edges:
			f.write(edge.replace(',', ', ') + '\n')

def read_graph(suffix):
	with open(cwd + 'nodes_' + suffix + '.txt') as f:
		nodes = []
		for line in f:
			nodes.append(line.rstrip())
	with open(cwd + 'edges_' + suffix + '.txt') as f:
		edges = []
		for line in f:
			edges.append(tuple(line.rstrip().split(', ')))
	return nodes, edges

if __name__ == '__main__':
	cwd = 'data/'
	cube = None

	if os.path.isfile(cwd + 'nodes_baseline.txt') and os.path.isfile(cwd + 'edges_baseline.txt'):
		authors, links = read_graph('baseline')
	else:
		with open('cube/models/step3.pkl', 'r') as f:
			cube = pickle.load(f)
		authors, links = cube.author0, DblpEval.author_links(cube, cube.author0)
		dump_graph(authors, links, 'baseline')
	plot(authors, links, 'baseline')

	if os.path.isfile(cwd + 'nodes_rl.txt') and os.path.isfile(cwd + 'edges_rl.txt'):
		authors, links = read_graph('rl')
	else:
		if cube is None:
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
			authors, reward, actions = agent.plan(sess)
		links = DblpEval.author_links(cube, authors)
		dump_graph(authors, links, 'rl')
	plot(authors, links, 'rl')
