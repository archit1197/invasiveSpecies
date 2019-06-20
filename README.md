## Introduction ##

Markov Decision Processes (MDP) are a fundamental mathematical abstraction used to model sequential decision making under uncertainty and are a
discrete model of discrete-time stochastic control and reinforcement learning
(RL). Particularly central to the real life applications of the modelling of real
life scenarios through MDPs is their planning, wherein we try to compute an
optimal policy that maps each state of an MDP to an action to be followed
at that state. The goal is to find an optimal policy which maximises the
utility of traversing the MDP. The modelling of the utility or the reward
model can be discounted, undiscounted, finite horizon, etc. which can be
chosen according to the practical application.

In the particular MDPs we study in this report, simulators used to replicate
the behaviour of the problem are of high accuracy, and hence are very expensive to compute. Consequently, the time required to solve these MDPs is
dominated by the number of calls to the simulator. A good MDP planning
algorithm in such domains should ideally minimise the number of calls of the
simulator yet terminate with a policy that is approximately optimal with
high probability. This is referred to being as PAC-RL.


