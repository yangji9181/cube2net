from config import *
from Environment import *
from PPO import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	agent = PPO(args, environment)
	with tf.Session() as sess:
		agent.train(sess)
		_, reward = agent.plan(sess)
		print('total reward: %f' % reward)
