from HD_eventprop import hd_eventprop
from tqdm import trange
import csv

# default parameters
params = {}
params["NUM_INPUT"] = 40
params["NUM_HIDDEN"] = 256
params["NUM_OUTPUT"] = 20
params["BATCH_SIZE"] = 128
params["INPUT_FRAME_TIMESTEP"] = 2
params["INPUT_SCALE"] = 0.008
params["NUM_EPOCH"] = 50
params["NUM_FRAMES"] = 80
params["verbose"] = False
params["lr"] = 0.01

#weights
params["hidden_w_mean"] = 0.5
params["hidden_w_sd"] = 4.0
params["output_w_mean"] = 0.5
params["output_w_sd"] = 1

file_path = "/its/home/ts468/data/rawHD/experimental_2/"

#hd_eventprop(params, file_path, True)

hidden_w_mean = []
hidden_w_sd = []
output_w_mean = []
output_w_sd = []

no_of_val = 11
div = 2.5

for i in range(no_of_val):
    hidden_w_mean.append(i/div)
    hidden_w_sd.append(i/div)
    output_w_mean.append(i/div)
    output_w_sd.append(i/div)

combinations = []

for hwm in hidden_w_mean:
    for hwsd in hidden_w_sd:
        for owm in output_w_mean:
            for owsd in output_w_sd:
                combinations.append([hwm, hwsd, owm, owsd])
                
                
file = open("grid_search_results", "w")
csv_writer = csv.writer(file, delimiter=",")

csv_writer.writerow(["hidden weight mean", "hidden weight sd", "output weight mean", "output weight sd", "accuracy"])
                
for i in trange(len(combinations)):
    #weights
    params["hidden_w_mean"] =   combinations[i][0]
    params["hidden_w_sd"] =     combinations[i][1]
    params["output_w_mean"] =   combinations[i][2]
    params["output_w_sd"] =     combinations[i][3]
    
    csv_writer.writerow([combinations[i][0],
                         combinations[i][1],
                         combinations[i][2],
                         combinations[i][3],
                         hd_eventprop(params, file_path, True)])

    file.flush()
    
    