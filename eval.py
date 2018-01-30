import pickle
from cube.utils import DblpEval
from config import *
from Environment import *
from PPO import *
from cube.cube_construction import DblpCube
from Cube import Cube

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	agent = PPO(args, environment)
	with tf.Session() as sess:
		agent.train(sess)
		authors, reward = agent.plan(sess)
		print('total reward: %f' % reward)

	with open('cube/models/step3.pkl', 'r') as f:
		cube = pickle.load(f)
	test = DblpEval(cube, authors)
	test.clusteringLINE()
