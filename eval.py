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
	os.environ['CUDA_VISIBLE_DEVICES'] = '7'
	with tf.device('/gpu:0'):
		agent = PPO(args, environment)
	with tf.Session(config=tf.ConfigProto(
			allow_soft_placement=True,
			gpu_options=tf.GPUOptions(
				per_process_gpu_memory_fraction=0.5,
				allow_growth=True))) as sess:
		agent.train(sess)
		authors, reward = agent.plan(sess)
		print('total reward: %f' % reward)

	with open('cube/models/step3.pkl', 'r') as f:
		cube = pickle.load(f)
	test = DblpEval(cube, authors, method='rl')
	test.clusteringLINE()
