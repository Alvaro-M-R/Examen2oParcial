# -*- coding: utf-8 -*-
"""
@author: Alvaro
"""
import random
from deap import creator, base, tools, algorithms
import numpy as np


random.seed(123)
NUM_CITIES = 5
MAX_X = 100
MAX_Y = 1000

POPULATION_SIZE = 300
MUTATION_RATIO = 0.1
CROSSOVER_RATIO = 0.5
NGEN = 40
cities = [      [0, 7, 9, 8, 20],
                [7, 0, 10, 4, 11],
                [9, 10, 0, 15, 5],
                [8, 4, 15, 0, 17],
                [20, 11, 5, 17, 0],]

def ini_salesman(container, num_cities):
    basic_plan = list(range(num_cities))
    random.shuffle(basic_plan)
    return container(basic_plan)
test = ini_salesman(list, 10)
def calc_distance(travel_plan, cities):
    dist = 0
    for i, e in enumerate(travel_plan):
        if i!= len(cities)-1:
            origin = cities[e]
            destination = cities[travel_plan[i+1]]
        else:
            origin = cities[e]
            destination = cities[travel_plan[0]]
        dist += origin.distance(destination)
    return dist,
def mutate_travel_plan(travel_plan):

    idx_1 = random.choice(list(range(len(travel_plan))))
    idx_2 = random.choice(list(range(len(travel_plan))))

    travel_plan[idx_1], travel_plan[idx_2] = travel_plan[idx_2], travel_plan[idx_1]
    
    return travel_plan,

def mate_travel_plans_single(tp_1, tp_2):
    
    N = len(tp_1)
    
    idx_1 = random.choice(list(range(N)))
    idx_2 = random.choice(list(range(N)))
    
    idx_start = min(idx_1, idx_2)
    idx_stop = max(idx_1, idx_2)
    
    if idx_start==idx_stop:
        if idx_start > 0:
            idx_start = idx_start-1
        else:
            idx_stop = idx_stop+1
    
    retain_sequence = tp_1[idx_start:idx_stop+1]
    substitute_values = [i for i in tp_2 if i not in retain_sequence]
    substitute_places = [i for i in list(range(N)) if i<idx_start or i>idx_stop]
    
    for i in substitute_places:
        tp_1[i] = substitute_values.pop(0)
    
    return tp_1
#mate_travel_plans_single([0,1,2,3,4], [3,2,1,0,4])

def mate_travel_plans(tp_1, tp_2):
    ind1 = mate_travel_plans_single(tp_1, tp_2)
    ind2 = mate_travel_plans_single(tp_1, tp_2)
    return ind1, ind2

creator.create("total_distance", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.total_distance) 
toolbox = base.Toolbox()
toolbox.register("individual", ini_salesman, creator.Individual, num_cities=NUM_CITIES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("travel_distance", calc_distance, cities=cities)
toolbox.register("mate", mate_travel_plans)
toolbox.register("mutate", mutate_travel_plan)

toolbox.register("select", tools.selTournament, tournsize=10)

population = toolbox.population(n=POPULATION_SIZE)
for gen in range(NGEN):
    print("Calculating generation {} of {}".format(gen+1,NGEN))

    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=MUTATION_RATIO)

    fits = toolbox.map(toolbox.travel_distance, offspring)

    population = toolbox.select(offspring, k=len(population))
    
winner = tools.selBest(population, k=1)
print(winner)