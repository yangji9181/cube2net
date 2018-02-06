import pickle
import time
import numpy as np
from collections import defaultdict
from cube.utils import DblpEval
from config import *
from Baseline import Baseline
from Cube import Cube
from cube.cube_construction import DblpCube

def print_cells(cube, cells):
	for t, v, y in cells:
		print(cube.topic_name[t][:10])
		print(cube.venue_name[v])
		print(y)



if __name__ == '__main__':
	with open('cube/models/step3.pkl', 'r') as f:
		cube = pickle.load(f)

	baseline = Baseline(args)
	state = baseline.initial_state()
	print(len(state))

	deepwalks, node2vecs = [], []
	for _ in range(args.num_exp):
		authors, reward, actions = baseline.random_baseline(state)
		print('random baseline: %f' % reward)
		cells = [baseline.cube.id_to_cell[id] for id in actions if id > -1]
		print_cells(cube, cells)
		test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='random')
		scores = test.evalAll(args.eval_dim, runs=1)
		deepwalks.append(scores[0][0])
		node2vecs.append(scores[0][1])
	print('deepwalk mean, std', np.mean(deepwalks, axis=0), np.std(deepwalks, axis=0))
	print('node2vec mean, std', np.mean(node2vecs, axis=0), np.std(node2vecs, axis=0))

	start = time.time()
	authors, reward, actions = baseline.greedy_baseline(state, args.baseline_candidate)
	print('greedy time %f s' % (time.time() - start))
	print('greedy baseline: %f' % reward)
	cells = [baseline.cube.id_to_cell[id] for id in actions if id > -1]
	print_cells(cube, cells)
	test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='greedy')
	print(test.evalAll(args.eval_dim, runs=1))
