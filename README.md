BrainDQN_Nature.py, BrainDQN_double.py, BrainDQN_doublePER.py, BrainDQN_doublePERdueling.py, BrainA3C.py, BrainPPO.py are all efficent RL methods to play Flappy Bird video game, respectively use Nature DQN, Double DQN, Double PER DQN, Double PER Dueling DQN, A3C, PPO, which can be found in my report.

BrainDQN_Nature_FTT.py is decomposing the FC layers using Tensor Train decomposition

BrainDQN_Nature_CTT.py is decomposing the Conv layers using Tensor Train decomposition

BrainDQN_Nature_TT.py is decomposing the whole network using method1 presented in my report

All of them above can be runned by FlappyBirdDQN.py

TT-cross.m is using DMRG cross of AMEN cross to make the TT-tensor through relu of sigmoid layer. Before running it, you should first download the TT Toolbox
