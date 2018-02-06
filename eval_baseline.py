import pickle
import time
import numpy as np
from collections import defaultdict
from cube.utils import DblpEval
from config import *
from Baseline import Baseline
from Cube import Cube
from cube.cube_construction import DblpCube



if __name__ == '__main__':
	with open('cube/models/step3.pkl', 'r') as f:
		cube = pickle.load(f)

	baseline = Baseline(args)
	state = baseline.initial_state()
	print(len(state))

	deepwalks, node2vecs = [], []
	for _ in range(args.num_exp):
		authors, reward = baseline.random_baseline(state)
		print('random baseline: %f' % reward)
		test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='random')
		scores = test.evalAll(args.eval_dim, runs=1)
		deepwalks.append(scores[0][0])
		node2vecs.append(scores[0][1])
	print('deepwalk mean, std', np.mean(deepwalks, axis=0), np.std(deepwalks, axis=0))
	print('node2vec mean, std', np.mean(node2vecs, axis=0), np.std(node2vecs, axis=0))

	deepwalks, node2vecs = [], []
	for _ in range(args.num_exp):
		start = time.time()
		authors, reward = baseline.greedy_baseline(state, args.baseline_candidate)
		print('greedy time %f s' % (time.time() - start))
		print('greedy baseline: %f' % reward)
		test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='greedy')
		scores = test.evalAll(args.eval_dim, runs=1)
		deepwalks.append(scores[0][0])
		node2vecs.append(scores[0][1])
	print('deepwalk mean, std', np.mean(deepwalks, axis=0), np.std(deepwalks, axis=0))
	print('node2vec mean, std', np.mean(node2vecs, axis=0), np.std(node2vecs, axis=0))

