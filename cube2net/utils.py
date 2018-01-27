import pickle
import snap
from collections import defaultdict
import evaluate
from cube_construction import DblpCube
from subprocess import call

class DblpCell(object):
	def __init__(self, year=-1, venue=-1, topic=-1, authors=set()):
		self.year = year
		self.venue = venue
		self.topic = topic
		self.authors = authors

	def getAuthors(self, cube):
		if len(self.authors) == 0:
			self.authors = cube.cell_year[self.year] & cube.cell_venue[self.venue] & cube.cell_topic[self.topic]
		return self.authors

	def getCells(self, author, cube):
		years = []
		venues = []
		topics = []
		for year in range(len(cube.year_name)):
			if author in cube.cell_year:
				years.append(year)
		for venue in range(lan(cube.venue_name)):
			if author in cube.cell_venue:
				venues.append(venue)
		for topic in range(len(cube.topic_name)):
			if author in cube.cell_topic:
				topics.append(topic)

		cells = []
		for year in years:
			for venue in venues:
				for topic in topics:
					cells.append(DblpCell(year=year, venue=venue, topic=topic))
		return cells


class DblpEval(object):
	def __init__(self, cube, authors):
		self.cube = cube
		self.edges = defaultdict(int)
		self.nodes = list(authors)
		authors = set(authors)
		for authors_list in self.cube.paper_author:
			coauthors = set(authors_list) & authors
			if len(coauthors) > 1:
				for i in coauthors:
					for j in coauthors:
						if i != j:
							self.edges[i+','+j] += 1
		
		label_file = '../clus_dblp/name-'+self.cube.params['label_type']+'.txt'
		self.names = []
		labels = []
		with open(label_file, 'r') as f:
			for line in f:
				tokens = line.split('\t')
				name = tokens[0].strip().replace('_', ' ')
				label = int(tokens[1].strip())
				self.names.append(name)
				labels.append(label)				

		k_true = len(set(labels))
		labelmap = list(set(labels))
		self.true = [[0]*len(self.names) for i in range(k_true)]
		for i in range(len(self.names)):
			self.true[labelmap.index(labels[i])][i] = 1


	def clusteringCNM(self):
		G = snap.TUNGraph.New()
		for i in range(len(self.nodes)):
			G.AddNode(i)
		for key in self.edges.keys():
			tokens = key.split(',')
			G.AddEdge(self.nodes.index(tokens[0]), self.nodes.index(tokens[1]))
		CmtyV = snap.TCnComV()
		modularity = snap.CommunityCNM(G, CmtyV)
		detected = []
		for cmty in CmtyV:
			detected.append(list(cmty))

		k_pred = len(detected)
		pred = [[0]*len(self.names) for i in range(k_pred)]
		i = 0
		for cmty in detected:
			for member in cmty:
				if self.nodes[member] in self.names:
					pred[i][self.names.index(self.nodes[member])] = 1
			i += 1

		print('f1: '+str(evaluate.f1_community(pred, self.true)))
		print('jc: '+str(evaluate.jc_community(pred, self.true)))
		print('nmi: '+str(evaluate.nmi_community(pred, self.true)))


	#mengxiong
	def clusteringLINE(self):
		'''
		need to define the network structure in line/node-a-0.txt and line/edge-aa-0.txt 
		according to the current format
		run the following system call
		after termination, output embeddings are available in line/output-a-0.txt
		'''

		with open('../line/node-a-0.txt', 'w') as nodef, open('../line/edge-aa-0.txt', 'w') as edgef:
			for i in range(len(self.nodes)):
				nodef.write(str(i)+'\n')
			for key in self.edges.keys():
				tokens = key.split(',')
				edgef.write(str(self.nodes.index(tokens[0]))+'\t'+str(self.nodes.index(tokens[1]))+'\t'+str(self.edges[key])+'\n')

		call('./embed -size 64 -iter 20', shell=True, cwd='../line')

		


		print('f1: '+str(evaluate.f1_community(pred, self.true)))
		print('jc: '+str(evaluate.jc_community(pred, self.true)))
		print('nmi: '+str(evaluate.nmi_community(pred, self.true)))

if __name__ == '__main__':
	with open('models/step3.pkl', 'r') as f:
		cube = pickle.load(f)
	test = DblpEval(cube, cube.author0)
	test.clusteringLINE()
	#test = DblpEval(cube, cube.author1)
	#test.clusteringCNM()
	#test = DblpEval(cube, cube.author2)
	#test.clusteringCNM()

