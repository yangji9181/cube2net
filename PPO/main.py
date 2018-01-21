from common.Buffer import *
from Environment import Environment
from PPO import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	agent = PPO(args, environment)
	with tf.Session() as sess:
		agent.train(sess)
		agent.plan(sess)
