import numpy as np
import util
from copy import deepcopy
from multiprocessing import *
from Cube import Cube
from config import *


class Baseline(object):
	def __init__(self, params):
		self.params = params
		self.cube = Cube.load_cube(args.cube_file)
		self.cell_embed = util.load_embed(params, self.cube)

	def load_embed(self):
		t_embed, v_embed = {}, {}
		with open(self.params.venue_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				v_embed[line[0]] = np.array(map(float, line[1].split()))
		with open(self.params.topic_file) as f:
			for line in f:
				line = line.rstrip().split('\t')
				t_embed[line[0]] = np.array(map(float, line[1].split()))
		cell_embed = []
		for id in range(len(self.cube.id_to_cell)):
			cell = self.cube.id_to_cell[id]
			concat = np.concatenate(t_embed[cell[0], v_embed[cell[1]]])
			cell_embed.append(np.insert(concat, 0, -0.5 + float(cell[2] - self.params.start_year)
			                            / float(self.params.end_year - self.params.start_year)))
		self.cell_embed = np.array(cell_embed)

	def initial_state(self):
		return self.cube.initial_state(self.params.test_file, self.params.low_limit, self.params.high_limit, self.params.debug)

	def random_baseline(self, state):
		actions = set(list(np.random.choice(len(self.cube.id_to_cell), self.params.trajectory_length, replace=False)))
		final = state | actions
		return self.cube.all_authors(final), self.cube.total_reward(final, self.params)

	def greedy_worker(self, state, candidates, num_worker, worker_id, queue):
		local_queue = []
		for idx, cell_id in enumerate(candidates):
			if cell_id not in state and idx % num_worker == worker_id:
				state_copy = deepcopy(state)
				state_copy.add(cell_id)
				local_queue.append((state_copy, self.cube.total_reward(state_copy, self.params)))
		if len(local_queue) > 0:
			queue.put(max(local_queue, key=lambda e: e[1]))
		else:
			queue.put((state, self.cube.total_reward(state, self.params)))

	def embedding_worker(self, state, candidates, num_worker, worker_id, queue):
		state_embed = np.array([self.cell_embed[id] for id in state])
		local_queue = []
		for idx, cell_id in enumerate(candidates):
			if cell_id not in state and idx % num_worker == worker_id:
				state_copy = deepcopy(state)
				state_copy.add(cell_id)
				local_queue.append((state_copy, np.amin(np.linalg.norm(self.cell_embed[cell_id] - state_embed, ord=2, axis=1))))
		if len(local_queue) > 0:
			queue.put(min(local_queue, key=lambda e: e[1]))
		else:
			queue.put((state, 0))

	def greedy_baseline(self, state, num_candidate, embedding=False):
		num_worker = self.params.num_process
		next = deepcopy(state)
		for i in range(self.params.trajectory_length):
			print('step %d' % i)
			candidates = list(np.random.choice(len(self.cube.id_to_cell), num_candidate, replace=False))
			queue = Queue()
			processes = []
			for id in range(num_worker):
				process = Process(target=self.embedding_worker if embedding else self.greedy_worker,
				                  args=(next, candidates, num_worker, id, queue))
				processes.append(process)
				process.start()
			nexts = []
			while not queue.empty():
				pair = queue.get()
				nexts.append(pair)
			for process in processes:
				process.join()
			next = max(nexts, key=lambda e: e[1])[0]
		return self.cube.all_authors(next), self.cube.total_reward(next, self.params)


if __name__ == '__main__':
	baseline = Baseline(args)
	state = baseline.initial_state()
	_, reward = baseline.random_baseline(state)
	print('random baseline: %f' % reward)
	_, reward = baseline.greedy_baseline(state, args.baseline_candidate, embedding=True)
	print('greedy embedding baseline: %f' % reward)
	_, reward = baseline.greedy_baseline(state, args.baseline_candidate)
	print('greedy baseline: %f' % reward)
