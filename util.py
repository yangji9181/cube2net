import numpy as np


def load_embed(params, cube):
	t_embed, v_embed = {}, {}
	embed_size = -1
	with open(params.venue_file) as f:
		for line in f:
			line = line.rstrip().split('\t')
			v_embed[int(line[0])] = np.array(map(float, line[1].split()))
			embed_size = len(np.array(map(float, line[1].split())))
	with open(params.topic_file) as f:
		for line in f:
			line = line.rstrip().split('\t')
			t_embed[int(line[0])] = np.array(map(float, line[1].split()))
	cell_embed = []
	for id in range(len(cube.id_to_cell)):
		cell = cube.id_to_cell[id]
		if cell[0] not in t_embed:
			t = np.zeros(embed_size)
		else:
			t = t_embed[cell[0]]
		if cell[1] not in v_embed:
			v = np.zeros(embed_size)
		else:
			v = v_embed[cell[1]]
		concat = np.concatenate((t, v), axis=0)
		cell_embed.append(np.insert(concat, 0, -0.5 + float(cell[2] - params.start_year)
		                            / float(params.end_year - params.start_year)))
	return np.array(cell_embed)
