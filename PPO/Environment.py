from common.Environment import *


class Environment(Base):
	def __init__(self, params):
		super(Environment, self).__init__(params)

	def trajectory_reward(self, state, actions):
		state = set([self.id_to_cell[id] for id in state])
		return self.cube.trajectory_reward(state, [self.id_to_cell[id] for id in actions], self.params.measure)
