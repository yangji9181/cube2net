import os
import argparse


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, default='DQN', help=None)
	parser.add_argument('--measure', type=str, default='transitivity', help=None)
	parser.add_argument('--hidden_dim', type=list, default=[64], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--embed_dim', type=int, default=65, help=None)
	parser.add_argument('--clip_epsilon', type=int, default=1e-1, help=None)
	parser.add_argument('--batch_size', type=int, default=2, help=None)
	parser.add_argument('--trajectory_length', type=int, default=2, help=None)
	parser.add_argument('--epoch', type=int, default=10, help=None)
	parser.add_argument('--k_step', type=int, default=10, help=None)
	parser.add_argument('--actor_step', type=int, default=1, help=None)
	parser.add_argument('--critic_step', type=int, default=1, help=None)
	parser.add_argument('--initial_state_size', type=int, default=1, help=None)
	parser.add_argument('--start_year', type=int, default=1954, help=None)
	parser.add_argument('--end_year', type=int, default=2018, help=None)
	return parser.parse_args()


def init_dir(args):
	args.data_dir = os.getcwd() + '/data/'
	args.venue_file = args.data_dir + 'venue.txt'
	args.content_file = args.data_dir + 'content.txt'
	args.year_file = args.data_dir + 'year.txt'
	args.cube_file = args.data_dir + 'cube.pkl'
	args.cell_file = args.data_dir + 'cell.txt'
	args.replay_buffer_file = args.data_dir + 'buffer.txt'

args = parse_args()
init_dir(args)
