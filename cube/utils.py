import pickle
import time
import shutil
from collections import defaultdict
import evaluate
from cube_construction import DblpCube
from subprocess import call
from sklearn.cluster import KMeans
import numpy as np
import os

class DblpEval(object):
	#label_type:
		#group: small set of 116 authors
		#label: large set of 4236 authors
	def __init__(self, cube, authors, links, label_type='label', method='unknown'):
		self.cube = cube
		self.label_type = label_type
		self.method = method
		self.edges = links
		self.nodes = list(authors)
		self.root = './'
		if not os.path.exists('clus_dblp'):
			self.root = '../'
		
		label_file = self.root + 'clus_dblp/name-' + label_type + '.txt'
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

		print('num nodes %d' % len(self.nodes))
		print('num edges %d' % len(self.edges))

	def enlargeGraph(self, cells=[]):
		for c in cells:
			self.nodes = list(set(self.nodes) | (self.cube.year_author[c[0]] & self.cube.venue_author[c[1]] & self.cube.topic_author[c[2]]))
			for key in self.cube.year_link[c[0]].keys():
				if key in self.cube.venue_link[c[1]] and key in self.cube.topic_link[c[2]]:
					value = min(self.cube.year_link[c[0]][key], self.cube.venue_link[c[1]][key], self.cube.topic_link[c[2]][key])
					self.edges[key] = max(self.edges[key], value)

	def writeGraph(self, format_='line'):
		print('writing graphs...')
		with open(self.root + 'cube/models/'+self.label_type + '_' + self.method + '_node.txt', 'w') as nodef, open(self.root + 'cube/models/'+self.label_type + '_' + self.method + '_edge.txt', 'w') as edgef:
			for i in range(len(self.nodes)):
				nodef.write(str(i)+'\n')
			for key in self.edges.keys():
				tokens = key.split(',')
				if format_ == 'line':
					edgef.write(str(self.nodes.index(tokens[0]))+'\t'+str(self.nodes.index(tokens[1]))+'\t'+str(self.edges[key])+'\n')
				elif format_ == 'deepwalk' or format_ == 'node2vec':
					edgef.write(str(self.nodes.index(tokens[0]))+' '+str(self.nodes.index(tokens[1]))+'\n')
		print('finished graph writing.')

	def embeddingLINE(self, embed_size):
		self.embed_method = 'line'
		self.writeGraph(format_='line')
		shutil.copyfile(self.root + 'cube/models/'+self.label_type + '_' + self.method + '_node.txt', self.root + 'line/node-a-0.txt')
		shutil.copyfile(self.root + 'cube/models/'+self.label_type + '_' + self.method + '_edge.txt', self.root + 'line/edge-aa-0.txt')

		call('./embed -size %d -iter 1000' % embed_size, shell=True, cwd=self.root+'line/')

		self.embed = np.zeros((len(self.names), embed_size))
		with open(self.root+'line/output-a-0.txt', 'r') as embf:
			for line in embf:
				tokens = line.split('\t')
				if self.nodes[int(tokens[0])] in self.names:
					self.embed[self.names.index(self.nodes[int(tokens[0])])] = np.array(list(map(float, tokens[1].strip().split(' '))))

	def embeddingDeepWalk(self, embed_size):
		self.embed_method = 'deepwalk'
		self.writeGraph(format_='deepwalk')
		shutil.copyfile(self.root + 'cube/models/'+self.label_type + '_' + self.method + '_edge.txt', self.root + 'deepwalk/edgelist.txt')

		call('deepwalk --format edgelist --input edgelist.txt --output embeddings_out.txt --representation-size %d' % embed_size, shell=True, cwd=self.root+'deepwalk/')
		self.embed = np.zeros((len(self.names), embed_size))
		with open(self.root+'deepwalk/embeddings_out.txt', 'r') as embf:
			for line in embf:
				tokens = line.strip().split(' ')
				if len(tokens) > 2:
					if self.nodes[int(tokens[0])] in self.names:
						self.embed[self.names.index(self.nodes[int(tokens[0])])] = np.array(list(map(float, tokens[1:])))

	def embeddingNode2Vec(self, embed_size):
		self.embed_method = 'node2vec'
		self.writeGraph(format_='node2vec')
		shutil.copyfile(self.root + 'cube/models/'+self.label_type + '_' + self.method + '_edge.txt', self.root + 'node2vec/edgelist.txt')

		call('python2 src/main.py --input edgelist.txt --output embeddings_out.txt --dimensions %d ' % embed_size, shell=True, cwd=self.root+'node2vec/')
		self.embed = np.zeros((len(self.names), embed_size))
		with open(self.root+'node2vec/embeddings_out.txt', 'r') as embf:
			for line in embf:
				tokens = line.strip().split(' ')
				if len(tokens) > 2:
					if self.nodes[int(tokens[0])] in self.names:
						self.embed[self.names.index(self.nodes[int(tokens[0])])] = np.array(list(map(float, tokens[1:])))

	def evalClustering(self):
		kmeans = KMeans(n_clusters=self.k_true).fit(self.embed)
		pred = [[0]*len(self.names) for i in range(self.k_true)]
		for i in range(len(self.names)):
			pred[kmeans.labels_[i]][i] = 1
		
		f1 = evaluate.f1_community(pred, self.true)
		jc = evaluate.jc_community(pred, self.true)
		nmi = evaluate.nmi_community(pred, self.true)
		print('f1: '+str(f1))
		print('jc: '+str(jc))
		print('nmi: '+str(nmi))

		return((f1, jc, nmi))

	def evalAll(self, embed_size, runs=1):
		u = np.ndarray(shape=(2, 3), dtype=np.float32)
		s = np.ndarray(shape=(2, 3), dtype=np.float32)

		f1 = []
		jc = []
		nmi = []
		for i in range(runs):
			start = time.time()
			self.embeddingDeepWalk(embed_size)
			print('deepwalk running time %f s' % (time.time() - start))
			t = self.evalClustering()
			f1.append(t[0])
			jc.append(t[1])
			nmi.append(t[2])
		u[0, :] = np.array([np.mean(f1), np.mean(jc), np.mean(nmi)])
		s[0, :] = np.array([np.std(f1), np.std(jc), np.std(nmi)])

		f1 = []
		jc = []
		nmi = []
		for i in range(runs):
			start = time.time()
			self.embeddingNode2Vec(embed_size)
			print('node2vec running time %f s' % (time.time() - start))
			t = self.evalClustering()
			f1.append(t[0])
			jc.append(t[1])
			nmi.append(t[2])
		u[1, :] = np.array([np.mean(f1), np.mean(jc), np.mean(nmi)])
		s[1, :] = np.array([np.std(f1), np.std(jc), np.std(nmi)])

		return((u, s))

	def cellGreedy(self, maxn=100, runs=5, basenodes=set()):
		res_u = np.ndarray(shape=(maxn, 3, 3), dtype=np.float32)
		res_s = np.ndarray(shape=(maxn, 3, 3), dtype=np.float32)
		celllist_file = open(self.root+'cube/models/celllist_'+str(len(basenodes))+'.txt', 'w')
		cs = []
		for c in range(maxn):
			overlap = 0
			for i in range(len(self.cube.year_author)):
				for j in range(len(self.cube.venue_author)):
					for k in range(len(self.cube.topic_author)):
						o = len(self.cube.year_author[i] & self.cube.venue_author[j] & self.cube.topic_author[k] & basenodes)
						if (o > overlap) and ((i, j, k) not in cs):
							overlap = o
							t = (i, j, k)
			if overlap == 0:
				celllist_file.write('Added %d cells, no more cell found.\n' %c)
				break
			cs.append(t)
			celllist_file.write('Adding cell %d with %d overlaps:\n' % (c, overlap))
			celllist_file.write('<'+str(self.cube.year_name[t[0]])+', '+str(self.cube.venue_name[t[1]])+', '+str(self.cube.topic_name[t[2]])+'>\n')
			self.enlargeGraph([t])
			t = self.evalAll(128, runs=runs)
			res_u[c,:,:] = t[0]
			res_s[c,:,:] = t[1]

		celllist_file.close()
		with open(self.root+'cube/models/perform_'+str(len(basenodes))+'.pkl', 'wb') as f:
			pickle.dump(res_u, f)
			pickle.dump(res_s, f)

	@staticmethod
	def author_links(cube, authors):
		links = defaultdict(int)
		for authors_list in cube.paper_author:
			coauthors = set(authors_list) & authors
			if len(coauthors) > 1:
				for i in coauthors:
					for j in coauthors:
						if i != j:
							links[i + ',' + j] += 1
		return links

if __name__ == '__main__':
	with open('models/step3.pkl', 'rb') as f:
		cube = pickle.load(f)
	with open('models/basenet.pkl', 'rb') as f:
		basenet = pickle.load(f)
	#test = DblpEval(cube=cube, authors=basenet['author0'], links=basenet['link0'], method='base0')
	#test.evalAll()
	#test = DblpEval(cube=cube, authors=basenet['author1'], links=basenet['link1'], method='base1')
	#test.evalAll()
	#test = DblpEval(cube=cube, authors=basenet['author2'], links=basenet['link2'], method='base2')
	#test.evalAll()

	test = DblpEval(cube=cube, authors=basenet['author0'], links=basenet['link0'], method='cellGreedy')
	test.cellGreedy(basenodes=basenet['author0'])
	test.cellGreedy(basenodes=basenet['author1'])
	test.cellGreedy(basenodes=basenet['author2'])

