import pickle


class DblpCube(object):
	pass

class Cube(object):
	def __init__(self, cube):
		topic_author, venue_author, year_author = {}, {}, {}
		topic_link, venue_link, year_link = {}, {}, {}
		for idx, authors in enumerate(cube.topic_author):
			topic_author[idx] = authors
		for idx, authors in enumerate(cube.venue_author):
			venue_author[idx] = authors
		with open('data/year_name.txt') as f:
			for idx, line in enumerate(f):
				year_author[int(line.rstrip())] = cube.year_author[idx]

		for idx, authors in enumerate(cube.topic_link):
			topic_link[idx] = set([tuple(pair.split(',')) for pair in authors.keys()])
		for idx, authors in enumerate(cube.venue_link):
			venue_link[idx] = set([tuple(pair.split(',')) for pair in authors.keys()])
		with open('data/year_name.txt') as f:
			for idx, line in enumerate(f):
				year_link[int(line.rstrip())] = set([tuple(pair.split(',')) for pair in cube.year_link[idx].keys()])

		self.id_to_cell = []
		self.id_to_author = []
		self.id_to_link = []
		for topic, author_t in topic_author.items():
			print(topic)
			for venue, author_v in venue_author.items():
				for year, author_y in year_author.items():
					author_c = set.intersection(author_t, author_v, author_y)
					if len(author_c) >= 100:
						cell = (topic, venue, year)
						self.id_to_cell.append(cell)
						self.id_to_author.append(author_c)
						link_c = set.intersection(topic_link[topic], venue_link[venue], year_link[year])
						self.id_to_link.append(link_c)


if __name__ == '__main__':
	with open('cube/models/step3.pkl', 'rb') as f:
		dblp = pickle.load(f)
	cube = Cube(dblp)
	print(len(cube.id_to_cell))
	pickle.dump(cube, open('data/cube.pkl', 'wb'))
