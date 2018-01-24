import numpy as np
from copy import deepcopy
from multiprocessing import *
from common.Cube import Cube
from common.config import *


class Baseline(object):
	def __init__(self, params):
		self.params = params
		self.cube = Cube.load_cube(args.cube_file)

	def initial_state(self):
		return set(list(np.random.choice(len(self.cube.id_to_cell), self.params.initial_state_size, replace=False)))

	def random_baseline(self, state):
		actions = set(list(np.random.choice(len(self.cube.id_to_cell), self.params.trajectory_length, replace=False)))
		final = state | actions
		return self.cube.total_reward([self.cube.id_to_cell[id] for id in final], self.params.measure)

	def greedy_worker(self, state, num_worker, worker_id, queue):
		local_queue = []
		for a in self.cube.id_to_cell:
			if a not in state and a % num_worker == worker_id:
				state_copy = deepcopy(state)
				state_copy.add(a)
				local_queue.append((state_copy, self.cube.total_reward([self.cube.id_to_cell[id] for id in state_copy], self.params.measure)))
		queue.put(max(local_queue, key=lambda e: e[1]))

	def greedy_baseline(self, state):
		num_worker = 4
		next = deepcopy(state)
		for _ in range(self.params.trajectory_length):
			queue = Queue()
			processes = []
			for id in range(num_worker):
				process = Process(target=self.greedy_worker, args=(next, num_worker, id, queue))
				processes.append(process)
				process.start()
			for process in processes:
				process.join()
			nexts = []
			while not queue.empty():
				pair = queue.get()
				nexts.append(pair)
			next = max(nexts, key=lambda e: e[1])[0]
		return self.cube.total_reward([self.cube.id_to_cell[id] for id in next], self.params.measure)


if __name__ == '__main__':
	baseline = Baseline(args)
	state = baseline.initial_state()
	# print('random baseline: %f' % baseline.random_baseline(state))
	print('greedy baseline: %f' % baseline.greedy_baseline(state))
