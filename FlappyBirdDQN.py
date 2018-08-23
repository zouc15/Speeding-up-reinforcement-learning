import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
#from BrainDQN_Nature_FTT import BrainDQN
#from BrainDQN_Nature_CTT import BrainDQN
#from BrainDQN_double import BrainDQN
#from BrainDQN_Nature_TT import BrainDQN
#from BrainDQN_doublePER import BrainDQN
#from BrainDQN_doublePERdueling import BrainDQN
#from BrainDQN_Nature_CP import BrainDQN

import numpy as np
import time
import matplotlib.pyplot as plt

# preprocess raw image to 80*80 gray image
def preprocess(observation):
	observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
	return np.reshape(observation,(80,80,1))

def playFlappyBird():
	# Step 1: init BrainDQN
	SCORE=[]
	actions = 2
	brain = BrainDQN(actions)
	# Step 2: init Flappy Bird Game
	flappyBird = game.GameState()
	# Step 3: play game
	# Step 3.1: obtain init state
	action0 = 0  # do nothing
	observation0, reward0, terminal, score = flappyBird.frame_step(action0)
	observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
	brain.setInitState(observation0)

	# Step 3.2: run the game
	timestep=0
	start1=time.time()
	while timestep <= 10000 and time.time()-start1 <= 6000:
		start = time.time()
		action = brain.getAction()
		nextObservation,reward,terminal,score = flappyBird.frame_step(action)
		if score != 'n':
			SCORE.append(score)
		nextObservation = preprocess(nextObservation)
		brain.setPerception(nextObservation,action,reward,terminal)
		timestep=timestep+1
		print(time.time()-start,'\n')
	#brain.plot_cost()
	result=set(SCORE)
	length=len(SCORE)
	y=[]
	for i in result:
		num = SCORE.count(i)
		y.append(num)
		print('the percent of score ' + str(i) + ' is ' + str(num/length))
	x=list(result)
	plt.bar(x,y)
	plt.show()

def main():
	playFlappyBird()

if __name__ == '__main__':
	main()
