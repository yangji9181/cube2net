import pickle
from collections import defaultdict
import evaluate
from cube_construction import DblpCube
from subprocess import call
from sklearn.cluster import KMeans
import numpy as np


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
		
		label_file = 'clus_dblp/name-'+self.cube.params['label_type']+'.txt'
		self.names = []
		labels = []
		with open(label_file, 'r') as f:
			for line in f:
				tokens = line.split('\t')
				name = tokens[0].strip().replace('_', ' ')
				label = int(tokens[1].strip())
				self.names.append(name)
				labels.append(label)				

		self.k_true = len(set(labels))
		labelmap = list(set(labels))
		self.true = [[0]*len(self.names) for i in range(self.k_true)]
		for i in range(len(self.names)):
			self.true[labelmap.index(labels[i])][i] = 1



	#mengxiong
	def clusteringLINE(self):
		'''
		need to define the network structure in line/node-a-0.txt and line/edge-aa-0.txt 
		according to the current format
		run the following system call
		after termination, output embeddings are available in line/output-a-0.txt
		'''

		with open('line/node-a-0.txt', 'w') as nodef, open('line/edge-aa-0.txt', 'w') as edgef:
			for i in range(len(self.nodes)):
				nodef.write(str(i)+'\n')
			for key in self.edges.keys():
				tokens = key.split(',')
				edgef.write(str(self.nodes.index(tokens[0]))+'\t'+str(self.nodes.index(tokens[1]))+'\t'+str(self.edges[key])+'\n')

		embed_size = 128
		call('./embed -size %d -iter 100' % embed_size, shell=True, cwd='line/')

		embed = np.ndarray(shape=(len(self.names), embed_size), dtype=np.float64)
		with open('line/output-a-0.txt', 'r') as embf:
			for line in embf:
				tokens = line.split('\t')
				if self.nodes[int(tokens[0])] in self.names:
					embed[self.names.index(self.nodes[int(tokens[0])]), :] = np.array(list(map(float, tokens[1].strip().split(' '))))

		kmeans = KMeans(n_clusters=self.k_true).fit(embed)
		pred = [[0]*len(self.names) for i in range(self.k_true)]
		for i in range(len(self.names)):
			pred[kmeans.labels_[i]][i] = 1
		

		print('f1: '+str(evaluate.f1_community(pred, self.true)))
		print('jc: '+str(evaluate.jc_community(pred, self.true)))
		print('nmi: '+str(evaluate.nmi_community(pred, self.true)))

if __name__ == '__main__':
	with open('models/step3.pkl', 'r') as f:
		cube = pickle.load(f)
	test = DblpEval(cube, cube.author0)
	test.clusteringLINE()
	test = DblpEval(cube, cube.author1)
	test.clusteringLINE()
	test = DblpEval(cube, cube.author2)
	test.clusteringLINE()

