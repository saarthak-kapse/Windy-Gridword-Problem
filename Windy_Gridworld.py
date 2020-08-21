import os
import numpy as np 
import sys 

class Windy_Gridworld():

	def __init__(self,number_of_rows=7,number_of_columns=10,start_state=[3,0],goal_state=[3,7],horizontal_wind_pattern=[],vertical_wind_pattern=[],vertical_stochasticity=False,horizontal_stochasticity=False):
		
		self.number_of_rows = number_of_rows
		self.number_of_columns = number_of_columns
		self.start_state = start_state
		self.goal_state = goal_state
		self.vertical_stochasticity = vertical_stochasticity 
		self.horizontal_stochasticity = horizontal_stochasticity

		if vertical_wind_pattern == []:
			self.vertical_wind_pattern = np.array([0]*number_of_columns)
		else:
			self.vertical_wind_pattern = np.array(vertical_wind_pattern)

		if horizontal_wind_pattern == []:
			self.horizontal_wind_pattern = np.array([0]*number_of_rows)
		else:
			self.horizontal_wind_pattern = np.array(horizontal_wind_pattern)

	def predict(self, current_state, current_action):

		# defined to take into consideration if start_state given is same as goal_state
		if current_state == self.goal_state:  
			return current_state,0

		next_state = [0,0]
		current_action_number_v = 0
		current_action_number_h = 0
		vertical_stochastic_number = 0
		horizontal_stochastic_number = 0

		if self.vertical_stochasticity is True:
			if self.vertical_wind_pattern[current_state[1]] != 0:
				vertical_stochastic_number = np.random.choice([-1,0,1])

		if self.horizontal_stochasticity is True:
			if self.horizontal_wind_pattern[current_state[0]] != 0:
				horizontal_stochastic_number = np.random.choice([-1,0,1])

		if current_action == 'up':
			current_action_number_v = -1
		elif current_action == 'down':
			current_action_number_v = 1
		elif current_action == 'left':
			current_action_number_h = -1
		elif current_action == 'right':
			current_action_number_h = 1
		elif current_action == 'up right':
			current_action_number_v = -1
			current_action_number_h = 1
		elif current_action == 'down right':
			current_action_number_v = 1
			current_action_number_h = 1
		elif current_action == 'up left':
			current_action_number_v = -1
			current_action_number_h = -1
		elif current_action == 'down left':
			current_action_number_v = 1
			current_action_number_h = -1

		next_state[0] = current_state[0] + self.vertical_wind_pattern[current_state[1]] \
						+ current_action_number_v + vertical_stochastic_number 
		next_state[1] = current_state[1] + self.horizontal_wind_pattern[current_state[0]] \
						+ current_action_number_h + horizontal_stochastic_number
		
		if next_state[0] < 0:
			next_state[0] = 0
		elif next_state[0] >= self.number_of_rows:
			next_state[0] = self.number_of_rows-1

		if next_state[1] < 0:
			next_state[1] = 0
		elif next_state[1] >= self.number_of_columns:
			next_state[1] = self.number_of_columns-1


		if next_state == self.goal_state:
			reward = 1  
		else :
			reward = -1

		return next_state,reward
