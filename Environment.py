import numpy as np


class Environment(object):
	def __init__(self, params):
		self.params = params
		self.load_cell()
		self.load_embed()
		self.load_replay_buffer()

	def load_cell(self):
		self.cell_to_id = {}
		self.id_to_cell = {}
		with open(self.params.cell_file) as f:
			for line in f:
				splits = line.rstrip().split('\t')
				id, venue, year = int(splits[0]), splits[1].split('|')[0], int(splits[1].split('|')[1])
				self.cell_to_id[(venue, year)] = id
				self.id_to_cell[id] = (venue, year)
		self.action_size = len(self.id_to_cell)

	def load_embed(self):
		v_embed, y_embed = {}, {}
		with open(self.params.venue_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				v_embed[line[0]] = np.array(map(float, line[1].split()))
		with open(self.params.year_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				y_embed[int(line[0])] = np.array(map(float, line[1].split()))

		cell_embed = []
		for id in range(len(self.id_to_cell)):
			cell = self.id_to_cell[id]
			cell_embed.append(np.concatenate((v_embed[cell[0]], y_embed[cell[1]])))
		self.cell_embed = np.array(cell_embed)

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
		return self.replay_buffer[np.random.choice(len(self.replay_buffer), self.params.batch_size)]

	def state_embed(self, state):
		return np.mean(self.cell_embed[state], axis=0)
