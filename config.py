import argparse
import os
import sys


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--debug', type=bool, default=False, help=None)
	parser.add_argument('--measure', type=str, default='transitivity', help=None)
	parser.add_argument('--hidden_dim', type=list, default=[256], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--embed_dim', type=int, default=129, help=None)
	parser.add_argument('--clip_epsilon', type=float, default=1e-1, help=None)
	parser.add_argument('--c_value', type=float, default=1.0, help='Coefficient for value function loss')
	parser.add_argument('--batch_size', type=int, default=4, help='Number of trajectories sampled')
	parser.add_argument('--trajectory_length', type=int, default=2, help=None)
	parser.add_argument('--epoch', type=int, default=4, help=None)
	parser.add_argument('--step', type=int, default=2, help=None)
	parser.add_argument('--intersect_threshold', type=int, default=100, help='High value may cause an error')
	parser.add_argument('--start_year', type=int, default=1954, help=None)
	parser.add_argument('--end_year', type=int, default=2018, help=None)
	parser.add_argument('--num_process', type=int, default=4, help='Number of subprocesses')
	parser.add_argument('--baseline_candidate', type=int, default=10, help='Number of candidates for baseline')
	return parser.parse_args()


def init_dir(args):
	if sys.platform == 'darwin':
		args.data_dir = os.getcwd() + '/data/'
	else:
		args.data_dir = '/shared/data/mliu60/Cube2Net/data/'
	args.venue_file = args.data_dir + 'venue.txt'
	args.topic_file = args.data_dir + 'topic.txt'
	args.content_file = args.data_dir + 'content.txt'
	args.year_file = args.data_dir + 'year.txt'
	args.cube_file = args.data_dir + 'cube.pkl'
	args.cell_file = args.data_dir + 'cell.txt'
	args.test_file = args.data_dir + 'label.txt'

args = parse_args()
init_dir(args)
