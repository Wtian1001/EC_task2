###############################################################################
# EvoMan FrameWork - Test Best Solution                                       #
# Test file to evaluate the best solution from one run using `evaluate()`     #
###############################################################################

import sys
from evoman.environment import Environment
from demo_controller import player_controller
import numpy as np
import pandas as pd
import os
import EA1_gen

# Choose the enemy
enemylist1 = [1, 2, 5]
enemylist2 = [4, 6, 7]
enemylist = enemylist2
loc = 29

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = f'EA1_enemylist2'
n_hidden_neurons = 10

# Load the best solution from a previous run (adjust file path as needed)
best_solution_file = f'{experiment_name}/{experiment_name}.csv'

# Read the best solution for a specific run (you can modify this to select the run you want)
best_solutions_df = pd.read_csv(best_solution_file)
best_solution_str = best_solutions_df.loc[loc, 'BEST SOL']  # Change the index to select a different run
print(f'Best solution location: {best_solutions_df.loc[loc]}')
best_solution = np.array(eval(best_solution_str))  # Convert string back to a numpy array

# initializes simulation in individual evolution mode, for single static enemy
env = Environment(experiment_name=experiment_name,
                  multiplemode="yes",
                  enemies=enemylist,
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    gain = p - e
    return f, gain

# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))

# uses evaluate function to test one individual
def evaluate_best_solution(env, best_sol):
    # Evaluate the individual (reshape into 2D array to pass as population of one)
    best_sol = np.array([best_sol])  # Convert the solution into a 2D array (1xN)
    return evaluate(best_sol)

result = evaluate_best_solution(env, best_solution)
gain = result[0][1]  # Get the gain from the evaluation result
print(f'Test: Gain = {gain}')