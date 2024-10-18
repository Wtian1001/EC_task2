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

# Choose the enemy
enemylist = [1, 2, 3, 4, 5, 6, 7, 8]
loc = 59

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

#write solution to txt file
with open(f'best_solution.txt', 'w') as f:
    for item in best_solution:
        f.write("%s\n" % item)

# initializes simulation in individual evolution mode, for single static enemy
env = Environment(experiment_name=experiment_name,
                  #multiplemode="yes",
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

play_life = []
enemy_life = []
for enemy in enemylist:
    env.update_parameter('enemies', [enemy])
    f,p,e,t = env.play(pcont=best_solution)
    print(f'enemy: {enemy} player life: {p}, enemy life: {e}')
    play_life.append(p)
    enemy_life.append(e)

# write csv
df = pd.DataFrame({'enemy': enemylist, 'player_life': play_life, 'enemy_life': enemy_life})
df.to_csv(f'competition_result.csv', index=False)
