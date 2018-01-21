from common.Buffer import *
from Environment import Environment
from DQN import *


if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	agent = DQN(args, environment)
	with tf.Session() as sess:
		agent.train(sess)
		agent.plan(sess)
