## Implementation of *cube2net: Efficient Quality Network Construction with Data Cube Organization* by Carl Yang, Mengxiong Liu, Frank He, Jian Peng, Jiawei Han. ICDM 2019 PhD Forum.

Please cite the following work if you find the code useful.

`
@inproceedings{yang2019cube2net,
	Author = {Yang, Carl and Liu, Mengxiong and He, Frank and Peng, Jian and Han, Jiawei},
	Booktitle = {ICDM},
	Title = {cube2net: efficient quality network construction with data cube organization},
	Year = {2019}
}
`
  

step 1: cell network construction (mengxiong)
based on current cell construction code, construct three networks, assume single hiararchy for each of them for now.
1. venue network based on venue names (in case venues are too many, do clustering first, it will also be useful when we later consider hiararchical cells)
2. year network
3. content network based on topic models

step 2: reinforcement learning state and value function design (mengxiong)
follow nips 2017 graph dqn work ([1])
1. implement the embedding of each dimension based on Eq 3 in [1], combine three dimensions and implement the value function based on Eq 4 in [1].

step 3: reinforcement learning algorithm (shibi)
1. design actions: add new cells: choose a dimension and a hiararchy to extend (how to explore efficiently?)
2. trade-off between exploit and exploration
3. algorithm: what kind of q-learning is more appropriate and why?

step 4; reward and evaluation (mengxiong)
use author clustering as the task for now, based on dblp labeled authors on the server
1. construct author network N based on chosen cells
2. evaluate chosen cells by evaluating a basic graph clustering algorithm like mincut on N
