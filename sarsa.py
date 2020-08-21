import os
import numpy as np 
import sys 

def epsilon_greedy(instance, epsilon=0.01):
	if np.random.choice([0,1], p = [epsilon, 1-epsilon]) == 0:
		action = np.random.randint(0, len(instance)-1)
	else:
		action_list = np.where(instance == np.max(instance))[0]
		action = np.random.choice(action_list)

	return action

class sarsa:
	def __init__(self, number_of_episodes, action_list, grid, alpha=0.01, epsilon=0.01,randomSeed=0):
		self.number_of_episodes = number_of_episodes
		self.grid = grid
		self.start_state = grid.start_state
		self.goal_state = grid.goal_state
		self. action_list = action_list
		self.alpha = alpha
		self.epsilon = epsilon
		self.randomSeed = randomSeed
		np.random.seed(self.randomSeed)

	def solve(self):

		num_to_direction_dict = {}
		for i in range(len(self.action_list)):
			num_to_direction_dict[i] = self.action_list[i]

		Q_function = np.zeros((self.grid.number_of_rows,self.grid.number_of_columns,len(self.action_list)))
		time_step_list = []
		time = 0
		for i in range(self.number_of_episodes):
			time_step_list.append(time)
			current_state = self.start_state
			current_action = epsilon_greedy(instance = Q_function[current_state[0],current_state[1],:], epsilon = self.epsilon)

			terminate = False
			while(terminate is False):
				time += 1
				current_action_word = num_to_direction_dict[current_action]
				next_state, reward = self.grid.predict(current_state, current_action_word)
				next_action = epsilon_greedy(instance = Q_function[next_state[0],next_state[1],:], epsilon = self.epsilon)
				Q_function[current_state[0],current_state[1],current_action] = Q_function[current_state[0],current_state[1],current_action] \
																			+ self.alpha*(reward + Q_function[next_state[0],next_state[1],next_action] \
																			- Q_function[current_state[0],current_state[1],current_action])
				
				current_state = next_state
				current_action = next_action

				if current_state == self.goal_state:
					terminate = True			

		return Q_function, time_step_list