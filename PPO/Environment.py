from common.Environment import *


class Environment(Base):
	def __init__(self, params):
		super(Environment, self).__init__(params)

	def terminal_action(self, action):
		return (action == len(self.id_to_cell) - 1) or (action == (None, -1))

	def initial_state(self):
		return np.random.choice(len(self.id_to_cell) - 1, self.params.initial_state_size, replace=False)

	def next_state(self, state, action):
		next = set(state)
		next.add(action)
		return np.array(list(next))

	def trajectory_reward(self, actions):
		if self.cube is None:
			self.cube = Cube.load_cube(self.params.cube_file)
		return self.cube.trajectory_reward([self.id_to_cell[id] for id in actions])
