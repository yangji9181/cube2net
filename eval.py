import pickle
import time
from cube.utils import DblpEval
from config import *
from Environment import *
from PPO import *
from cube.cube_construction import DblpCube
from Cube import Cube

if __name__ == '__main__':
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
		start = time.time()
		authors, reward = agent.plan(sess)
		end = time.time()
		print('total reward: %f' % reward)
		print('time %f s' % (end - start))

	with open('cube/models/step3.pkl', 'r') as f:
		cube = pickle.load(f)

	test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='rl')
	print(test.evalAll(args.eval_dim, runs=3))
