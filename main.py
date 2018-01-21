from common.Buffer import *

if __name__ == '__main__':
	if args.model == 'DQN':
		from DQN.Environment import Environment
		from DQN.DQN import *
	else:
		from PPO.Environment import Environment
		from PPO.PPO import *
	environment = Environment(args)
	tf.reset_default_graph()
	agent = eval(args.model)(args, environment)
	with tf.Session() as sess:
		agent.train(sess)
		reward = agent.plan(sess)
		print('total reward: %f' % reward)
