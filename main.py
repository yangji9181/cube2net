from Agent import *
from Buffer import *
from Environment import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	agent = Agent(args, environment)
	with tf.Session() as sess:
		agent.train(sess)
		agent.plan(sess)
