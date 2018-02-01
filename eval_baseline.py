import pickle
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
	test = DblpEval(cube, authors, label_type=label_type, method='random')
	test.writeGraph()
	test.clusteringLINE()

	# authors, reward = baseline.greedy_baseline(state, args.baseline_candidate, embedding=True)
	# print('greedy embedding baseline: %f' % reward)
	# test = DblpEval(cube, authors, label_type=label_type, method='embedding')
	# test.writeGraph()
	# test.clusteringLINE()

	authors, reward = baseline.greedy_baseline(state, args.baseline_candidate)
	print('greedy baseline: %f' % reward)
	test = DblpEval(cube, authors, label_type=label_type, method='greedy')
	test.writeGraph()
	test.clusteringLINE()
