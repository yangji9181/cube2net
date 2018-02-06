import pickle
import time
from cube.utils import DblpEval
from config import *
from Environment import *
from PPO import *
from cube.cube_construction import DblpCube
from Cube import Cube

def print_cells(cube, cells):
	for t, v, y in cells:
		print(cube.topic_name[t][:10])
		print(cube.venue_name[v])
		print(y)


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
		start = time.time()
		agent.train(sess)
		end = time.time()
		print('training time: %f s' % (end - start))
		start = time.time()
		authors, reward, actions = agent.plan(sess)
		end = time.time()
		print('total reward: %f' % reward)
		print('time %f s' % (end - start))

	with open('cube/models/step3.pkl', 'r') as f:
		cube = pickle.load(f)

	cells = [environment.cube.id_to_cell[id] for id in actions]
	print_cells(cube, cells)

	test = DblpEval(cube, authors, DblpEval.author_links(cube, authors), label_type=label_type, method='rl')
	print(test.evalAll(args.eval_dim, runs=1))
