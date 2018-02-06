import argparse
import os


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--device_id', type=int, default=7, help=None)
	parser.add_argument('--debug', type=bool, default=False, help=None)
	parser.add_argument('--transitivity_c', type=float, default=0.1, help='Reward coefficient')
	parser.add_argument('--connectivity_c', type=float, default=0.9, help='Reward coefficient')
	parser.add_argument('--hidden_dim', type=list, default=[256], help=None)
	parser.add_argument('--learning_rate', type=float, default=1e-3, help=None)
	parser.add_argument('--embed_dim', type=int, default=129, help=None)
	parser.add_argument('--clip_epsilon', type=float, default=1e-1, help=None)
	parser.add_argument('--c_value', type=float, default=1.0, help='Coefficient for value function loss')
	parser.add_argument('--batch_size', type=int, default=40, help='Number of trajectories sampled')
	parser.add_argument('--trajectory_length', type=int, default=20, help=None)
	parser.add_argument('--epoch', type=int, default=80, help=None)
	parser.add_argument('--outer_step', type=int, default=80, help='Number of rounds of mini batch SGD per epoch')
	parser.add_argument('--step', type=int, default=2, help=None)
	parser.add_argument('--low_limit', type=int, default=200, help=None)
	parser.add_argument('--high_limit', type=int, default=201, help=None)
	parser.add_argument('--start_year', type=int, default=1954, help=None)
	parser.add_argument('--end_year', type=int, default=2018, help=None)
	parser.add_argument('--num_process', type=int, default=2, help='Number of subprocesses')
	parser.add_argument('--init_state_limit', type=int, default=1, help=None)
	parser.add_argument('--baseline_candidate', type=int, default=2000, help='Number of candidates for baseline')
	parser.add_argument('--eval_dim', type=int, default=2048, help='Dimension for evaluation framework')
	return parser.parse_args()


def init_dir(args):
	args.data_dir = os.getcwd() + '/data/'
	args.venue_file = args.data_dir + 'venue.txt'
	args.topic_file = args.data_dir + 'topic.txt'
	args.content_file = args.data_dir + 'content.txt'
	args.year_file = args.data_dir + 'year.txt'
	args.cube_file = args.data_dir + 'cube.pkl'
	args.cell_file = args.data_dir + 'cell.txt'
	args.test_file = args.data_dir + label_type + '.txt'

label_type = 'label'
args = parse_args()
init_dir(args)
