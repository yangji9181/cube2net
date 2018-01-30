import numpy as np


def load_embed(params, cube):
	t_embed, v_embed = {}, {}
	with open(params.venue_file) as f:
		for line in f:
			line = line.rstrip().split('\t')
			v_embed[int(line[0])] = np.array(map(float, line[1].split()))
	with open(params.topic_file) as f:
		for line in f:
			line = line.rstrip().split('\t')
			t_embed[int(line[0])] = np.array(map(float, line[1].split()))
	cell_embed = []
	for id in range(len(cube.id_to_cell)):
		cell = cube.id_to_cell[id]
		concat = np.concatenate((t_embed[cell[0]], v_embed[cell[1]]), axis=0)
		cell_embed.append(np.insert(concat, 0, -0.5 + float(cell[2] - params.start_year)
		                            / float(params.end_year - params.start_year)))
	return np.array(cell_embed)
