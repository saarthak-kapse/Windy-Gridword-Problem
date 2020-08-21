import os
import numpy as np 
import sys 
from Windy_Gridworld import Windy_Gridworld
from sarsa import sarsa
import matplotlib.pyplot as plt
from operator import add

grid = Windy_Gridworld(number_of_rows=7,number_of_columns=10,start_state=[3,0],goal_state=[3,7],vertical_wind_pattern=[0,0,0,1,1,1,2,2,1,0],vertical_stochasticity=True)
action_list = ['up','down','left','right','up right','up left','down right','down left']
number_of_episodes = 1500

time_step_list = [0]*number_of_episodes
for i in range(10):
	solver = sarsa(number_of_episodes, action_list, grid, alpha=0.5, epsilon=0.1, randomSeed=i)
	Q_function, time_step_list_temp = solver.solve()
	time_step_list = list(map(add, time_step_list_temp, time_step_list))

time_step_list[:] = [x / (i+1) for x in time_step_list]

print('Task4 Fastest Route Time:',(time_step_list[-1]-time_step_list[-10])/10)
plt.plot(time_step_list,np.arange(len(time_step_list)))
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.savefig('Plots/task4.png')