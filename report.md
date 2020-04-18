# Report : Continous Control

### Index
1. Problem statement
2. Policy gradient and Actor-Critic Approach
3. DDPG Algorithm
4. Implementation details
5. Experiment and results
6. Next Steps


### 1. Problem Statement

##### Environment :
In this environment, a double-jointed arm can move to target locations. A reward of `+0.1` is provided for each step that the agent's hand is in the goal location. I other words, this consists of a moving location or 'orb' , and a controllable agent
The observation space consists of `33` variables corresponding to position, rotation, velocity, and angular velocities of the arm.  Each action is a vector with four numbers, corresponding to torque applicable to two joints.  Every entry in the action vector must be a number between `-1` and `1`.
##### Agent :
Its a moveable, double-jointed arm with 4 degrees of freedom or in other words, 4 actions.

##### Challenge : 
Design and develop an agent that should be able to follow the orb or target location while its moving. In mathematical terms. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.


### 2. Policy Gradient and Actor-Critic Approach

#### Policy Gradient
It offers a solution to the problem of having continous action space, which are difficult to solve with popular methods like _sarsa_ or DQN. *Value-Based Methods* like DQN or TD(0) obtain an optimal policy $\pi_*$ by trying to estimate the optimal action-value function, *Policy-Based Methods* directly learn the optimal policy. This simplification allows these type of methods to handle handle either discrete or continuous actions. They also have to advantage of combining Monte-Carlo returns with TD returns (Actor-Critic for example)
Policy based methods offer advatages like :
* Better conversion properties. This comes from the fact that value based methods tend to oscillate as it chooses different actions while training
* Works better in high dimensional space having continous action space, in contrast to DQN
* Can learn stochastic policies as well, DQN cannot

#### Actor Critic Approach
This approaches uses two function approximator namely an Actor and a Critic.
* Actor learns a stochastic policy by using a Monte-Carlo approach. It decides the probability of action(s) to take.  
* Critic learns the value function. It is used to tell whether the actions taken by the Actor are good or not.
Proess is as follows

1. Observe state $s$ from environment 
2. Use Actor to get action distribution $\pi(a|s;\theta_\pi)$. Stochasticially select one action stochastically and feed back to the environment.  
3. Observe next state $s'$ and reward $r$.  
4. Use the tuple $(s, a, r, s')$ for the TD estimate $y=r + \gamma V(s'; \theta_v)$
5. Use critic loss as $L=(y - V(s;\theta_v)^2$ for training the Critic network to minimize it 
6. Calculate the advantage $A(s,a) = r + \gamma V(s'; \theta_v) - V(s; \theta_v)$ and use it to traing the Actor


### 3. DDPG


**Deep Deterministic Policy Gradient (DDPG) [Lillicrap et al., 2016]** is an off policy algorithm that combines the above explained Actor-Critic approach with the well known DQN. Below is a brief description of  
- The actor function $\mu(s;\theta_\mu)$ gives the current policy. It maps states to continuous deterministic actions
- The critic $Q(s,a;\theta_q)$ on used to calculate action values and is learned using the Bellman equation
- Use of replay buffer - tuples of $(s, a, r, s')$ are stored and then batches are sampled to train the networks. This reduced the issue of network being trained from correlated tuples,which arises from sequentially exploring the environment
- Use of *target networks*, similar to DQN, allowed stable learning. A regular but softly (explained below) updated copy of the network is used as taget, for both Actor and Critic. They are used to calculate target values

Soft update allows only a fraction of the current/local network to change target network weights. This allows for continous target value improvments, unlike DQN where target network is update after every few thousand iterations with full replication:

$ \theta' \leftarrow \tau \theta + (1-\tau)\theta'$ with $\tau \ll 1$

Batch normalization is used while training to learn the importance of entire state space. Exploration in continous space is handled ny adding a noise distribution process $N$ an exploration policy $\mu'$ is constructed:

$\mu'(s_t) = \mu(s_t;\theta_{\mu,t})+N$

The DDPG pseudocode:


<img src='img/Pseudocode.png' height='200'/>



### 4. Implementation details

For details of the implementation please refer to the class definition files and DDPG_Control notebook. Brief description of each of the files is given below
* noise.py : Contains the class definition of noise function. It follows OU noise distribution without decay.
* model.py : Contains the definitions of Actor, Critic and Replay buffer class. All these classes are instantiated in the DDPG class, which implements the DDPG algorithm explained above
* ddpg.py : Contains the class definition of DDPG agent. It instantiates a pair , each of Actor and Critic classes, to be used as local and target network. The algorithm details is primarily defined in _act_ and _learn_ function of the class
* DDPG_Continous_Control.ipynb : Contains details of overall implementation, from setting up the environment to episodic learning of solving the reacher environment using DDPG agent.


### 5. Experiment and Results

It was noted that batch size is an important factor in learning. Steadily increasing batch size reduced episodes taken to learn.
This is a score graph for a batch size of 2048. The target was achieved in 113 episodes (100 episodes of average rewards 30+), but the rewards started hitting 30 as early as episode 10, owing to a larger batch size

<img src='img/score_graph.png' height='100'>


And this one is for a batch size of 512

<img src='img/score_512.png' height='100'>


Other parameters chosen are
Network parameters : A typical size of 128 and 256 is chosen as layer size.
Tau = 0.01 -> I chose a slightly higher update rate for faster learning. This led to slightly turbulent episodes, but it did the job
max_steps = 5000 ; approximate maximum time steps to get a total reward of 30
buffer_size = 10000000

Parameters below have assigned their typical values 
actor learning rate = 0.0001
critic learning rate = 0.001
gamma = 0.9


### 6. Next Steps

There are many things that can be tried
- A different algorithm like A3C
- Different hyperparameters like making tau 1e-5 but much larger batch size, like 8192 or more