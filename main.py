from Agent import *
from Buffer import *
from Environment import *

if __name__ == '__main__':
	environment = Environment(args)
	tf.reset_default_graph()
	agent = Agent(args, environment)
	agent.train()
	agent.play()
