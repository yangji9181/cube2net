import pickle
import time
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

	authors, reward = baseline.random_baseline(state)
	print('random baseline: %f' % reward)
	test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='random')
	print(test.evalAll(args.eval_dim, runs=3))

	# authors, reward = baseline.greedy_baseline(state, args.baseline_candidate, embedding=True)
	# print('embedding baseline: %f' % reward)
	# test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='embedding')
	# print(test.evalAll(args.eval_dim, runs=3))

	start = time.time()
	authors, reward = baseline.greedy_baseline(state, args.baseline_candidate)
	end = time.time()
	print('greedy baseline: %f' % reward)
	print('time %f s' % (end - start))
	test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='greedy')
	print(test.evalAll(args.eval_dim, runs=3))
