import pickle
import sys
import os
import re
import json
from time import gmtime, strftime
from collections import defaultdict
from gensim import corpora, models
from subprocess import call

class DblpCube(object):
	def __init__(self, params):
		self.params = params
		self.year_name = []
		self.year_author = []
		self.year_link = []
		self.venue_name = []
		self.venue_author = []
		self.venue_link = []
		self.topic_name = [[] for x in range(self.params['num_topics'])]
		self.topic_author = [set() for x in range(self.params['num_topics'])]
		self.topic_link = [defaultdict(int) for x in range(self.params['num_topics'])]
		self.paper_author = []

	def step1(self):
		basenet = {}
		basenet['author0'] = set()
		basenet['author1'] = set()
		basenet['author2'] = set()
		basenet['link0'] = defaultdict(int)
		basenet['link1'] = defaultdict(int)
		basenet['link2'] = defaultdict(int)
		all_authors = set()
		all_links = defaultdict(int)

		# input v
		v = set()
		with open(self.params['author_file']+self.params['label_type']+'.txt', 'r') as f:
			for line in f:
				name = line.strip().replace('_', ' ')
				v.add(name)
		print('#v: '+str(len(v)))

		# first scan on dblp data, record author1, print some stats
		for file_name in self.params['dblp_files']:
			with open(file_name, 'r') as f:
				for line in f:
					p = json.loads(line)
					if ('id' not in p) \
					  or ('authors' not in p) \
					  or ('venue' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['venue'])) \
					  or ('year' not in p) \
					  or (p['year'] < 1900) \
					  or ('title' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['title'])) \
					  or ('abstract' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['abstract'])):
						continue
					
					vs = set(p['authors']) & v
					if len(vs) > 0:
						for vv in vs:
							basenet['author0'].add(vv)
						for name in p['authors']:
							basenet['author1'].add(name)

					for name1 in p['authors']:
						all_authors.add(name1)
						for name2 in p['authors']:
							if name1 != name2:
								all_links[name1+','+name2] += 1

		print('#author0: '+str(len(basenet['author0'])))
		print('#author1: '+str(len(basenet['author1'])))

		with open('../clus_dblp/name-label.txt', 'r') as f1, open('../clus_dblp/name-label_.txt', 'w') as f2:
			for line in f1:
				tokens = line.split('\t')
				if tokens[0].strip().replace('_', ' ') in basenet['author0']:
					f2.write(line)

		# second scan on dblp data, record author2
		for file_name in self.params['dblp_files']:
			with open(file_name, 'r') as f:
				for line in f:
					p = json.loads(line)
					if ('id' not in p) \
					  or ('authors' not in p) \
					  or ('venue' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['venue'])) \
					  or ('year' not in p) \
					  or (p['year'] < 1900) \
					  or ('title' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['title'])) \
					  or ('abstract' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['abstract'])):
						continue
					
					if len(set(p['authors']) & basenet['author1']) > 0:
						for name in p['authors']:
							basenet['author2'].add(name)
		print('#author2: '+str(len(basenet['author2'])))

		# third scan on dblp data, record year, venue and content of papers including author2
		count = 0
		valid = 0
		content_file = open(self.params['content_file'], 'w')
		for file_name in self.params['dblp_files']:
			with open(file_name, 'r') as f:
				for line in f:
					count += 1
					p = json.loads(line)
					if ('id' not in p) \
					  or ('authors' not in p) \
					  or (len(set(p['authors']) & basenet['author2']) == 0) \
					  or ('venue' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['venue'])) \
					  or ('year' not in p) \
					  or (p['year'] < 1900) \
					  or ('title' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['title'])) \
					  or ('abstract' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['abstract'])):
						continue
					
					p_id = valid
					valid += 1

					# convert a paper to set of authors
					self.paper_author.append(set(p['authors']))

					# record paper contents and write all of them into a single file for auto-phrase mining
					content = p['title'].replace('\n','')+'\n'+p['abstract'].replace('\n','')+'\n'
					content_file.write(content.encode('utf-8'))

					# construct venue and year cells
					if p['venue'].lower() not in self.venue_name:
						self.venue_name.append(p['venue'].lower())
						self.venue_author.append(set())
						self.venue_link.append(defaultdict(int))
					self.venue_author[self.venue_name.index(p['venue'].lower())] |= self.paper_author[p_id]
					for a1 in self.paper_author[p_id]:
						for a2 in self.paper_author[p_id]:
							if a1 != a2:
								self.venue_link[self.venue_name.index(p['venue'].lower())][a1+','+a2] += 1
					if p['year'] not in self.year_name:
						self.year_name.append(p['year'])
						self.year_author.append(set())
						self.year_link.append(defaultdict(int))
					self.year_author[self.year_name.index(p['year'])] |= self.paper_author[p_id]
					for a1 in self.paper_author[p_id]:
						for a2 in self.paper_author[p_id]:
							if a1 != a2:
								self.year_link[self.year_name.index(p['year'])][a1+','+a2] += 1

		content_file.close()

		for authors_list in self.paper_author:
			coauthors = authors_list & basenet['author0']
			if len(coauthors) > 1:
				for i in coauthors:
					for j in coauthors:
						if i != j:
							basenet['link0'][i+','+j] += 1
			coauthors = authors_list & basenet['author1']
			if len(coauthors) > 1:
				for i in coauthors:
					for j in coauthors:
						if i != j:
							basenet['link1'][i+','+j] += 1
			coauthors = authors_list & basenet['author2']
			if len(coauthors) > 1:
				for i in coauthors:
					for j in coauthors:
						if i != j:
							basenet['link2'][i+','+j] += 1

		print('#link0: '+str(len(basenet['link0'])))
		print('#link1: '+str(len(basenet['link1'])))
		print('#link2: '+str(len(basenet['link2'])))

		with open('models/basenet.pkl', 'wb') as f:
			pickle.dump(basenet, f)

		with open('models/step1.pkl', 'wb') as f:
			pickle.dump(self, f)
		print('step1: finished processing '+str(valid)+'/'+str(count)+' papers.')
		print('#venue: '+str(len(self.venue_name)))
		print('#year: '+str(len(self.year_name)))
		print('#paper: '+str(len(self.paper_author)))
		print("#authors: %d, #links: %d " % (len(all_authors), len(all_links)))

	def step2(self):
		if not os.path.exists('models/segmentation.txt'):
			call('./phrasal_segmentation.sh', shell=True, cwd='../AutoPhrase')
		texts = []
		line_num = 0
		content = []
		tag_beg = '<phrase>'
		tag_end = '</phrase>'
		with open('models/segmentation.txt', 'r') as f:
			for line in f:
				while line.find(tag_beg) >= 0:
					beg = line.find(tag_beg)
					end = line.find(tag_end)+len(tag_end)
					content.append(line[beg:end].replace(tag_beg, '').replace(tag_end, '').lower())
					line = line[:beg] + line[end:]
				if line_num % 2 == 1:
					texts.append(content)
					content = []
				line_num += 1
				#if line_num % 20000 == 0:
				#		print("step2: "+strftime("%Y-%m-%d %H:%M:%S", gmtime())+': processing paper '+str(line_num//2))

		print("lda: constructing dictionary")
		dictionary = corpora.Dictionary(texts)
		print("lda: constructing doc-phrase matrix")
		corpus = [dictionary.doc2bow(text) for text in texts]
		print("lda: computing model")
		if not os.path.exists('models/ldamodel.pkl'):
			ldamodel = models.ldamodel.LdaModel(corpus, num_topics=self.params['num_topics'], id2word = dictionary, passes=20)
			with open('models/ldamodel.pkl', 'wb') as f:
				pickle.dump(ldamodel, f)
		else:
			with open('models/ldamodel.pkl', 'rb') as f:
				ldamodel = pickle.load(f)
		print("lda: saving topical phrases")
		for i in range(self.params['num_topics']):
			self.topic_name[i] = ldamodel.show_topic(i, topn=100)
		with open(self.params['topic_file'], 'w') as f:
			f.write(str(ldamodel.print_topics(num_topics=-1, num_words=10)))
		print('lda: finished.')

		counter = 0
		for paper in corpus:
			topics = ldamodel.get_document_topics(paper, minimum_probability=1e-4)
			topics.sort(key=lambda tup: tup[1], reverse=True)
			if len(topics) >= 1:
				self.topic_author[topics[0][0]] |= self.paper_author[counter]
				for a1 in self.paper_author[counter]:
						for a2 in self.paper_author[counter]:
							if a1 != a2:
								self.topic_link[topics[0][0]][a1+','+a2] += 1
			#if len(topics) >= 2:
				#self.cell_content_two[topics[1][0]].append(counter)
			#if len(topics) >= 3:
				#self.cell_content_three[topics[2][0]].append(counter)
			counter += 1

		with open('models/step2.pkl', 'wb') as f:
			pickle.dump(self, f)
		print('step2: finished processing '+str(counter)+' papers.')

	def step3(self):
		print('step3: writing network files.')
		num_node = 0
		num_edge = 0
		with open('models/year_name.txt', 'w') as namef, open('models/year_node.txt', 'w') as nodef, open('models/year_link.txt', 'w') as linkf:
			for year in self.year_name:
				namef.write(str(year)+'\n')
				nodef.write(str(self.year_name.index(year))+'\n')
				num_node += 1
				for year_c in self.year_name:
					if abs(int(year) - int(year_c)) == 1:
						linkf.write(str(self.year_name.index(year))+'\t'+str(self.year_name.index(year_c))+'\t1\n')
						num_edge += 1
		print('step3: finished year network files with '+str(num_node)+' nodes and '+str(num_edge)+' edges.')

		num_node = 0
		num_edge = 0
		with open('models/venue_name.txt', 'w') as namef, open('models/venue_node.txt', 'w') as nodef, open('models/venue_link.txt', 'w') as linkf:
			for venue in self.venue_name:
				namef.write(''.join(venue.split())+'\n')
				nodef.write(str(self.venue_name.index(venue))+'\n')
				num_node += 1
				for venue_c in self.venue_name:
					if venue != venue_c:
						same = len(set(venue.split()) & set(venue_c.split()))
						if same > 0:
							linkf.write(str(self.venue_name.index(venue))+'\t'+str(self.venue_name.index(venue_c))+'\t'+str(same)+'\n')
							num_edge += 1
		print('step3: finished venue network files with '+str(num_node)+' nodes and '+str(num_edge)+' edges.')

		num_node = 0
		num_edge = 0
		with open('models/topic_name.txt', 'w') as namef, open('models/topic_node.txt', 'w') as nodef, open('models/topic_link.txt', 'w') as linkf:
			for ind in range(len(self.topic_name)):
				namef.write(str(self.topic_name[ind])+'\n')
				nodef.write(str(ind)+'\n')
				num_node += 1
				for ind_c in range(len(self.topic_name)):
					if ind != ind_c:
						words = map(lambda x: x[0], self.topic_name[ind])
						words_c = map(lambda x: x[0], self.topic_name[ind_c])
						same = len(set(words) & set(words_c))
						if same > 0:
							linkf.write(str(ind)+'\t'+str(ind_c)+'\t'+str(same)+'\n')
							num_edge += 1
		print('step3: finished topic network files with '+str(num_node)+' nodes and '+str(num_edge)+' edges.')

		self.venue_name
		self.year_name
		self.topic_name
		with open('models/step3.pkl', 'wb') as f:
			pickle.dump(self, f)

if __name__ == '__main__':
	params = {}
	params['dblp_files'] = ['../dblp-ref/dblp-ref-0.json', '../dblp-ref/dblp-ref-1.json', '../dblp-ref/dblp-ref-2.json', '../dblp-ref/dblp-ref-3.json']
	params['author_file'] = '../clus_dblp/vocab-'
	params['label_type'] = 'label'
	params['content_file'] = 'models/content_file.txt'
	params['topic_file'] = 'models/topic_file.txt'
	params['num_topics'] = 100
	if not os.path.exists('models/step1.pkl'):
		cube = DblpCube(params)
		cube.step1()
	elif not os.path.exists('models/step2.pkl'):
		with open('models/step1.pkl', 'rb') as f:
			cube = pickle.load(f)
		cube.step2()
	elif not os.path.exists('models/step3.pkl'):
		with open('models/step2.pkl', 'rb') as f:
			cube = pickle.load(f)
		cube.step3()
	else:
		print('all 3 steps have finished.')


