from common.Environment import *


class Environment(Base):
	def __init__(self, params):
		super(Environment, self).__init__(params)
		self.load_replay_buffer()

	def load_replay_buffer(self):
		replay_buffer = []
		with open(self.params.replay_buffer_file) as f:
			for line in f:
				splits = line.rstrip().split('\t')
				state = np.array(list(map(lambda cell: int(cell), splits[0].split(','))))
				action = int(splits[1])
				next = np.array(list(map(lambda cell: int(cell), splits[2].split(','))))
				reward = float(splits[3])
				replay_buffer.append((state, action, next, reward))
		self.replay_buffer = np.array(replay_buffer)

	def sample(self):
		return self.replay_buffer[np.random.choice(len(self.replay_buffer), self.params.batch_size, replace=False)]
