import pickle
import sys
import os
import re
import json
from time import gmtime, strftime
from collections import defaultdict
from gensim import corpora, models

class DblpCube(object):
	def __init__(self, params):
		self.params = params
		self.cell_year_one = defaultdict(list)
		self.cell_year_three = defaultdict(list)
		self.cell_year_ten = defaultdict(list)
		self.cell_venue = defaultdict(list)
		self.cell_content_one = [[] for x in range(self.params['num_topics'])]
		self.cell_content_two = [[] for x in range(self.params['num_topics'])]
		self.cell_content_three = [[] for x in range(self.params['num_topics'])]
		self.topic_words = [[] for x in range(self.params['num_topics'])]
		self.paper_id = []
		self.paper_author = []

	def step1(self):
		count = 0
		valid = 0
		content_file = open(self.params['content_file'], 'w')
		for file_name in self.params['input_files']:
			with open(file_name, 'r') as f:
				for line in f:
					count += 1
					p = json.loads(line)
					if ('id' not in p) \
					  or ('authors' not in p) \
					  or (len(p['authors']) < 2) \
					  or ('venue' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['venue'])) \
					  or ('year' not in p) \
					  or (p['year'] < 1900) \
					  or ('title' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['title'])) \
					  or ('abstract' not in p) \
					  or (not re.match("^[\w\s,.:?-]+$", p['abstract'])):
						continue
					
					self.paper_id.append(p['id'])
					p_id = valid
					valid += 1

					# convert a paper to list of authors
					self.paper_author.append(p['authors'])

					# record paper contents and write all of them into a single file for auto-phrase mining
					content = p['title'].replace('\n','')+'\n'+p['abstract'].replace('\n','')+'\n'
					content_file.write(content.encode('utf-8'))

					# construct venue and year cells
					self.cell_venue[p['venue']].append(p_id)
					self.cell_year_one[p['year']].append(p_id)
					year_three = p['year'] // 3 * 3
					self.cell_year_three[year_three].append(p_id)
					year_ten = p['year'] // 10 * 10
					self.cell_year_ten[year_ten].append(p_id)

					if valid % 10000 == 0:
						print("step1: "+strftime("%Y-%m-%d %H:%M:%S", gmtime())+': processing paper '+str(valid))
						print(len(self.cell_venue), len(self.cell_year_one), len(self.cell_year_three), len(self.cell_year_ten))

		content_file.close()
		with open('step1.pkl', 'wb') as f:
			pickle.dump(self, f)
		print('step1: finished processing '+str(valid)+'/'+str(count)+' papers.')

	def step2(self):
		texts = []
		line_num = 0
		content = []
		tag_beg = '<phrase>'
		tag_end = '</phrase>'
		with open('segmentation.txt', 'r') as f:
			for line in f:
				while line.find(tag_beg) >= 0:
					beg = line.find(tag_beg)
					end = line.find(tag_end)+len(tag_end)
					content.append(line[beg:end].replace(tag_beg, '').replace(tag_end, ''))
					line = line[:beg] + line[end:]
				if line_num % 2 == 1:
					texts.append(content)
					content = []
				line_num += 1
				if line_num % 20000 == 0:
						print("step2: "+strftime("%Y-%m-%d %H:%M:%S", gmtime())+': processing paper '+str(line_num//2))

		print("lda: constructing dictionary")
		dictionary = corpora.Dictionary(texts)
		print("lda: constructing doc-phrase matrix")
		corpus = [dictionary.doc2bow(text) for text in texts]
		print("lda: computing model")
		ldamodel = models.ldamodel.LdaModel(corpus, num_topics=self.params['num_topics'], id2word = dictionary, passes=20)
		print("lda: saving topical phrases")
		self.topic_words = [[] for x in range(self.params['num_topics'])]
		for i in range(self.params['num_topics']):
			self.topic_words[i] = ldamodel.get_topic_terms(i, topn=10)
		with open(self.params['topic_file'], 'w') as f:
			f.write(str(ldamodel.print_topics(num_topics=-1, num_words=10)))
		print('lda: finished.')

		counter = 0
		for paper in corpus:
			topics = ldamodel.get_document_topics(paper, minimum_probability=1e-4)
			topics.sort(key=lambda tup: tup[1], reverse=True)
			if len(topics) >= 1:
				self.cell_content_one[topics[0][0]].append(counter)
			if len(topics) >= 2:
				self.cell_content_two[topics[1][0]].append(counter)
			if len(topics) >= 3:
				self.cell_content_three[topics[2][0]].append(counter)
			counter += 1

		with open('step2.pkl', 'wb') as f:
			pickle.dump(self, f)
		print('step2: finished processing '+str(counter)+' papers.')

	def step3(self):
		with open('year_network.txt', 'w') as networkf:
			with open('year_node.txt', 'w') as nodef:
				for year in self.cell_year_one.keys():
					nodef.write(str(year))
					for year_c in self.cell_year_one.keys():
						if year != year_c and abs(year-year_c)<=3:
							diff = abs(year-year_c)
							networkf.write(str(year)+'\t'+str(year_c)+'\t'+str(4-diff))

		with open('venue_network.txt', 'w') as networkf:
			with open('venue_node.txt', 'w') as nodef:
				for venue in self.cell_venue.keys():
					nodef.write(''.join(venue.split()))
					for venue_c in self.cell_venue.keys():
						if venue != venue_c:
							same = len(set(venue.split()) & set(venue_c.split()))
							if same > 0:
								networkf.write(''.join(venue.split())+'\t'+''.join(venue_c.split())+'\t'+str(same))

		with open('content_network.txt', 'w') as networkf:
			with open('content_node.txt', 'w') as nodef:
				for ind in range(len(self.topic_words)):
					nodef.write(str(ind))
					for ind_c in range(len(self.topic_words)):
						if ind != ind_c:
							words = map(lambda x: x[0], self.topic_words[ind])
							words_c = map(lambda x: x[0], self.topic_words[ind_c])
							same = len(set(words) & set(words_c))
							if same > 0:
								networkf.write(str(ind)+'\t'+str(ind_c)+'\t'+stc(same))

if __name__ == '__main__':
	params = {}
	params['input_files'] = ['dblp-ref/dblp-ref-0.json', 'dblp-ref/dblp-ref-1.json', 'dblp-ref/dblp-ref-2.json', 'dblp-ref/dblp-ref-3.json']
	#params['input_files'] = ['dblp-ref/dblp-ref-3.json']
	params['content_file'] = 'content_file.txt'
	params['topic_file'] = 'topic_file.txt'
	params['num_topics'] = 10
	if not os.path.exists('step1.pkl'):
		cube = DblpCube(params)
		cube.step1()
	elif not os.path.exists('step2.pkl'):
		with open('step1.pkl', 'rb') as f:
			cube = pickle.load(f)
		cube.step2()
	else:
		with open('step2.pkl', 'rb') as f:
			cube = pickle.load(f)
		cube.step3()


