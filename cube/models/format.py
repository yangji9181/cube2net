import shutil
from subprocess import call
from multiprocessing import *


dimension = ['topic', 'venue']
embed_size = 64

def deepwalk_thread(prefix):
	call('deepwalk --format edgelist --input edgelist.txt --output %s_out.txt --representation-size %d' % (prefix, embed_size), shell=True, cwd='../../deepwalk/')

def node2vec_thread(prefix):
	call('python2 src/main.py --input edgelist.txt --output %s_out.txt --dimensions %d ' % (prefix, embed_size), shell=True, cwd='../../node2vec/')


if __name__ == '__main__':
	for prefix in dimension:
		with open(prefix + '_link.txt') as fr:
			with open(prefix + '_edgelist.txt', 'w') as fw:
				for line in fr:
					splits = line.rstrip().split('\t')
					fw.write(splits[0] + ' ' + splits[1] + '\n')
		shutil.copyfile(prefix + '_edgelist.txt', '../../deepwalk/edgelist.txt')
		shutil.copyfile(prefix + '_edgelist.txt', '../../node2vec/edgelist.txt')

		processes = [Process(target=deepwalk_thread, args=(prefix,)), Process(target=node2vec_thread, args=(prefix,))]
		for process in processes:
			process.start()
		for process in processes:
			process.join()
