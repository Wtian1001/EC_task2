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

start_loc = 0

headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = f'EA1_enemylist1'
n_hidden_neurons = 10

# Load the best solution from a previous run (adjust file path as needed)
best_solution_file = f'{experiment_name}/{experiment_name}.csv'

# Read the best solution for a specific run (you can modify this to select the run you want)
best_solutions_df = pd.read_csv(best_solution_file)
#best_solution_str = best_solutions_df.loc[loc, 'BEST SOL']  # Change the index to select a different run
#print(f'Best solution location: {best_solutions_df.loc[loc]}')
#best_solution = np.array(eval(best_solution_str))  # Convert string back to a numpy array

# initializes simulation in individual evolution mode, for single static enemy
env = Environment(experiment_name=experiment_name,
                  enemies=[1,2,3,4,5,6,7,8],
                  multiplemode="yes",
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

gains = []
best_gens = []

for i in range(10):
    # Read the best solution for a specific run (you can modify this to select the run you want)
    #best_solutions_df = pd.read_csv(best_solution_file)
    for j in range(0,59):
        best_gain = -100
        loc = start_loc + j
        best_solution_str = best_solutions_df.loc[loc, 'BEST SOL']
        best_solution = np.array(eval(best_solution_str))  # Convert string back to a numpy array
        result = evaluate_best_solution(env, best_solution)
        temp_gain = result[0][1]
        if temp_gain >= best_gain:
            best_gain = temp_gain
            best_loc = loc
            best_gen = j

    print(f'Best solution location: {best_loc}')
    print(f'Test {i+1}: Gain = {best_gain}')
    gains.append(best_gain)
    best_gens.append(best_gen)
    start_loc = start_loc + 60



# Save results to a file
'''test_results_df = pd.DataFrame({"Test Run": list(range(1, 11)), "Gain": gains})
test_results_df.to_csv(f'{experiment_name}/test_solution.csv', index=False)

print(f'Test results saved to {experiment_name}/test_solution.csv')'''