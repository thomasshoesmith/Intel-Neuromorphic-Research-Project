import copy
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import sem

#export CUDA_PATH=/usr/local/cuda
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from tqdm import trange

from HD_eventprop import hd_eventprop

initial_pop = [0.0, 3.5, 3.0, 1.5]
pop_size = 10
pop = np.random.rand(pop_size,4) * 5
epochs = 10
pop_range = (0, 5)

pop[0] = initial_pop

def mutate(genotype, rate = .1):
    for i in range(len(genotype)):
        if random.random() <= rate:
          genotype[i] = random.random() * pop_range[1] #TODO
    return genotype

def crossover(genotype1, genotype2, p_crossover = .5):
    for i,gene in enumerate(genotype2):
      if random.random() <= p_crossover:
        genotype1[i] = genotype2[i]
    return genotype1

# constants
params = {}
params["NUM_INPUT"] = 40
params["NUM_HIDDEN"] = 256
params["NUM_OUTPUT"] = 20
params["BATCH_SIZE"] = 128
params["INPUT_FRAME_TIMESTEP"] = 20
params["INPUT_SCALE"] = 0.00099 #0.008
params["NUM_EPOCH"] = 80
params["NUM_FRAMES"] = 80
params["verbose"] = False
params["debug"] = False
params["lr"] = 0.01
params["dt"] = 1

params["reg_lambda_lower"] = 1e-11 
params["reg_lambda_upper"] = 1e-11
params["reg_nu_upper"] = 20

#weights
params["hidden_w_mean"] = 0.0 #0.5
params["hidden_w_sd"] = 3.5 #4.0
params["output_w_mean"] = 3.0
params["output_w_sd"] = 1.5 

file = open("grid_search_results.csv", "w")
csv_writer = csv.writer(file, delimiter=",")
csv_writer.writerow(["hidden weight mean", "hidden weight sd", "output weight mean", "output weight sd", "accuracy", "epoch"])

file.flush()

epochs = 50
num_individuals = 50
num_items = 10
k = 3
crossover_value = 0.5
mutationRate = 0.1
  
initial_pop = [0.0, 3.5, 3.0, 1.5]
num_individuals = pop_size = 10
pop = np.random.rand(pop_size,4) * 5
pop[0] = initial_pop

genotypes = pop
fitness = np.zeros(num_individuals)


x = []
y_best = []
y_mean = []
y_worst = []

for epoch in trange(epochs):

  id1 = np.random.randint(num_individuals) # Pick one individual at random, i.e. genotype  G1  at position  x1
  id2 = np.random.randint(id1 +1, id1 + k) % num_individuals # Pick a second individual  G2  in the local neighbourhood of the first, i.e., pick a competitor from the local neighbourhood in the range  x1+1  to  x1+k  (start with  k=2 ) 5.
  
  #weights
  params["hidden_w_mean"] = pop[id1, 0]
  params["hidden_w_sd"] =   pop[id1, 1]
  params["output_w_mean"] = pop[id1, 2]
  params["output_w_sd"] =   pop[id1, 3]
  geno1 = hd_eventprop(params)
  
  #weights
  params["hidden_w_mean"] = pop[id1, 0]
  params["hidden_w_sd"] =   pop[id1, 1]
  params["output_w_mean"] = pop[id1, 2]
  params["output_w_sd"] =   pop[id1, 3]
  geno2 = hd_eventprop(params)
  
  fitness[id1] = geno1
  fitness[id2] = geno2
  
  max_index = fitness.argmax()
  
  csv_writer.writerow([pop[max_index, 0],
                       pop[max_index, 1],
                       pop[max_index, 2],
                       pop[max_index, 3],
                       fitness[max_index],
                       epoch])
  
  file.flush()

  if (geno1 > geno2):
    genotypes[id2] = copy.deepcopy(genotypes[id1]) # Replace L with W
    genotypes[id1] = mutate(crossover(copy.deepcopy(genotypes[id1]), copy.deepcopy(genotypes[id2]), crossover_value), mutationRate)
  else:
    genotypes[id1] = copy.deepcopy(genotypes[id2]) # Replace L with W
    genotypes[id2] = mutate(crossover(copy.deepcopy(genotypes[id2]), copy.deepcopy(genotypes[id1]), crossover_value), mutationRate)

  y_best.append(np.amax(fitness))
  y_worst.append(np.amin(fitness))
  y_mean.append((sum(fitness)/len(fitness)))
  x.append(epoch)