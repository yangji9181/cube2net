import numpy as np
from copy import deepcopy
from common.config import *
from common.Cube import Cube


class Buffer(object):
	def __init__(self, params, cube):
		self.params = params
		self.cube = cube

	def generate_replay_buffer(self):
		with open(self.params.replay_buffer_file, 'w') as f:
			for num_cell in range(10, 200, 1):
				state, action, next, reward = self.generate_one(num_cell)
				f.write(','.join(map(lambda cell: str(self.cube.cell_to_id[cell]), state)) + '\t' + \
				        str(self.cube.cell_to_id[action]) + '\t' + \
				        ','.join(map(lambda cell: str(self.cube.cell_to_id[cell]), next)) + '\t' + \
				        str(reward) + '\n')

	def generate_one(self, num_cell):
		state = set(self.cube.id_to_cell[id] for id in np.random.choice(self.cube.num_cell, num_cell))
		action = self.cube.id_to_cell[np.random.choice(self.cube.num_cell, 1)[0]]
		next = deepcopy(state)
		next.add(action)
		reward = self.reward(state, action)
		return state, action, next, reward

	# state is a set of cells, action is a single cell
	def reward(self, state, action):
		return self.cube.reward(state, action, args.measure)

if __name__ == '__main__':
	cube = Cube.load_cube(args.cube_file)
	buffer = Buffer(args, cube)
	buffer.generate_replay_buffer()
