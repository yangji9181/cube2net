import argparse
import os
import sys


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='PPO', help=None)
	parser.add_argument('--measure', type=str, default='transitivity', help=None)
	parser.add_argument('--hidden_dim', type=list, default=[64], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--embed_dim', type=int, default=65, help=None)
	parser.add_argument('--clip_epsilon', type=float, default=1e-1, help=None)
	parser.add_argument('--c_value', type=float, default=1.0, help='Coefficient for value function loss')
	parser.add_argument('--batch_size', type=int, default=4, help='Number of trajectories sampled')
	parser.add_argument('--trajectory_length', type=int, default=20, help=None)
	parser.add_argument('--epoch', type=int, default=10, help=None)
	parser.add_argument('--k_step', type=int, default=10, help=None)
	parser.add_argument('--step', type=int, default=2, help=None)
	parser.add_argument('--initial_state_size', type=int, default=1, help=None)
	parser.add_argument('--intersect_threshold', type=int, default=100, help=None)
	parser.add_argument('--start_year', type=int, default=1954, help=None)
	parser.add_argument('--end_year', type=int, default=2018, help=None)
	parser.add_argument('--num_process', type=int, default=8, help='Number of subprocesses')
	return parser.parse_args()


def init_dir(args):
	if sys.platform == 'darwin':
		args.data_dir = os.getcwd() + '/data/'
	else:
		args.data_dir = '/shared/data/mliu60/Cube2Net/data/'
	args.venue_file = args.data_dir + 'venue.txt'
	args.content_file = args.data_dir + 'content.txt'
	args.year_file = args.data_dir + 'year.txt'
	args.cube_file = args.data_dir + 'cube.pkl'
	args.cell_file = args.data_dir + 'cell.txt'
	args.replay_buffer_file = args.data_dir + 'buffer.txt'
	args.test_file = args.data_dir + 'name-label.txt'

args = parse_args()
init_dir(args)
