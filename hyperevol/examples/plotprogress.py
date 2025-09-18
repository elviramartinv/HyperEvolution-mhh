# def read_fitness(output_dir, fitness_key="fitness"):
#     samples = os.path.join(output_dir, 'samples')
#     wild_card_path = os.path.join(samples, '*', 'score.json')
#     number_samples = len(glob.glob(wild_card_path))
#     score_dicts = []
#     for number in range(number_samples):
#         path = os.path.join(samples, str(number), 'score.json')
#         score_dict = read_json_cfg(path)
#         score_dicts.append(score_dict)
#     scores = [score_dict[fitness_key] for score_dict in score_dicts]
#     return scores

import os
import glob
import json
from statistics import median

def read_json_cfg(path):
    """ Reads the json info from a given path

    Parameters:
    ----------
    path : str
        Path to the .json file

    Returns:
    --------
    info : dict
        The json dict that was loaded
    """
    with open(path, 'rt') as jsonFile:
        info = json.load(jsonFile)
    return info

#foldername = "pso_23_newreweight_newbu"
foldername = "pso_23_newreweight_new"
#foldername = "pso_23_newreweight_new_good"
#foldername = "pso_23_newreweight_newbu2"

wildcardpath = foldername+"/previous_files/iteration_*"
maxiteration=len(glob.glob(wildcardpath))
scores = {}
fitness = {}
avgchi2 = {}
avgstat = {}
absolutbest=None
bestscores = []
absolutbest_fitness = -1
average_avgchi2 = []
average_avgstat = []
average_scores = []
average_fitness = []
max_avgchi2 = []
max_avgstat = []
max_scores = []
max_fitness = []
min_avgchi2 = []
min_avgstat = []
min_scores = []
min_fitness = []
for i in range(maxiteration):
    scores[i] = []
    fitness[i] = []
    avgchi2[i] = []
    avgstat[i] = []
    for r in glob.glob(os.path.join(foldername+"/previous_files/","iteration_{}".format(i),'*/score.json')):
        score_dict = read_json_cfg(r)
        if absolutbest_fitness is -1 or score_dict['fitness']<absolutbest_fitness:
            absolutbest_fitness=score_dict['fitness']
            absolutbest=r
        bestscores.append([score_dict['fitness'], r])
        scores[i].append(score_dict['score'])
        fitness[i].append(score_dict['fitness'])
        avgchi2[i].append(score_dict['avgchi2'])
        avgstat[i].append(score_dict['avgstat'])
    average_avgchi2.append(median(avgchi2[i]))
    average_avgstat.append(median(avgstat[i]))
    average_fitness.append(median(fitness[i]))
    average_scores.append(median(scores[i]))
    max_avgchi2.append(max(avgchi2[i]))
    max_avgstat.append(max(avgstat[i]))
    max_fitness.append(min(fitness[i]))
    max_scores.append(max(scores[i]))
    min_avgchi2.append(min(avgchi2[i]))
    min_avgstat.append(min(avgstat[i]))
    min_fitness.append(max(fitness[i]))
    min_scores.append(min(scores[i]))

print("best with:",absolutbest)
print("honorable mentions")
bestscores = sorted(bestscores, key = lambda x: x[0])
for i in range(10):
    print(bestscores[i+1])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
y = []
x = []
for i in range(maxiteration):
    y += scores[i]
    x += list(i*np.ones(len(scores[i])))
plt.scatter(x, y, c=x, alpha=0.05, cmap='viridis_r')
plt.plot(average_scores, color='black', label='average/iteration', linewidth=3)
plt.plot(max_scores, color='red', label='max score/iteratiom', linewidth=3)
plt.plot(min_scores, color='blue',  label='min score/iteration', linewidth=3)
#plt.yscale("log")
#plt.ylim(median(average_scores)-np.std(y)/2)
plt.ylabel("score")
plt.xlabel("iteration")
plt.legend()
plt.savefig("score.png")
plt.close()

y = []
x = []
for i in range(maxiteration):
    y += fitness[i]
    x += list(i*np.ones(len(fitness[i])))
plt.scatter(x, y, c=x, alpha=0.05, cmap='viridis_r')
plt.plot(average_fitness, color='black')
plt.plot(min_fitness, color='blue')
plt.plot(max_fitness, color='red')
#plt.yscale("log")
plt.ylabel("fitness")
plt.xlabel("iteration")
#plt.ylim(median(average_fitness)/100, median(average_fitness)*100)
plt.savefig("fitness.png")
plt.close()

y = []
x = []
for i in range(maxiteration):
    y += avgstat[i]
    x += list(i*np.ones(len(avgstat[i])))
plt.scatter(x, y, c=x, alpha=0.05, cmap='viridis_r')
plt.ylabel("relative stat error")
plt.xlabel("iteration")
plt.plot(average_avgstat, color='black', label='average/iteration', linewidth=3)
plt.plot(min_avgstat, color='blue',label='min error/iteration', linewidth=3)
plt.plot(max_avgstat, color='red',label='max error/iteration', linewidth=3)
plt.legend()
plt.yscale("log")
plt.ylim(1, 50*median(average_avgstat)+np.std(y)/2)
plt.savefig("avgstat.png")
plt.close()


y = []
x = []
for i in range(maxiteration):
    y += avgchi2[i]
    x += list(i*np.ones(len(avgchi2[i])))
plt.plot(average_avgchi2, color='black')
plt.plot(min_avgchi2, color='blue')
plt.plot(max_avgchi2, color='red')
plt.ylabel("$\chi^2/ndf$")
plt.xlabel("iteration")
plt.scatter(x, y, c=x, alpha=0.05, cmap='viridis_r')
plt.savefig("avgchi2.png")
plt.close()
