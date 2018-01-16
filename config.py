import argparse


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--hidden_dim', type=list, default=[128], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--embed_dim', type=int, default=64, help=None)
	parser.add_argument('--batch_size', type=int, default=10, help=None)
	parser.add_argument('--epoch', type=int, default=1, help=None)
	parser.add_argument('--k_step', type=int, default=1, help=None)
	return parser.parse_args()


def init_dir(args):
	args.data_dir = 'data/'
	args.venue_file = args.data_dir + 'venue.txt'
	args.content_file = args.data_dir + 'content.txt'
	args.year_file = args.data_dir + 'year.txt'
	args.cube_file = args.data_dir + 'cube.pkl'
	args.cell_file = args.data_dir + 'cell.txt'
	args.replay_buffer_file = args.data_dir + 'buffer.txt'

args = parse_args()
init_dir(args)
