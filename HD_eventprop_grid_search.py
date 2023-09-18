from HD_eventprop_real_time import hd_eventprop
from tqdm import trange
import csv
import os

# default parameters
params = {}
params["NUM_INPUT"] = 40
params["NUM_HIDDEN"] = 256
params["NUM_OUTPUT"] = 20
params["BATCH_SIZE"] = 128
params["INPUT_FRAME_TIMESTEP"] = 20
params["INPUT_SCALE"] = 0.0009 #0.008
params["NUM_EPOCH"] = 50
params["NUM_FRAMES"] = 80
params["verbose"] = False
params["lr"] = 0.01
params["dt"] = 1

#weights
params["hidden_w_mean"] = 0.5
params["hidden_w_sd"] = 4.0
params["output_w_mean"] = 0.5
params["output_w_sd"] = 1

file_path = os.path.expanduser("~/data/rawHD/experimental_2/")

#hd_eventprop(params, file_path, True)

hidden_w_mean = []
hidden_w_sd = []
output_w_mean = []
output_w_sd = []

range_of_val = 4
no_of_val = 5#8
div =  1 #no_of_val / range_of_val

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


print(hidden_w_mean)      
print(hidden_w_sd)        
print(output_w_mean)        
print(output_w_sd)            
        
file = open("grid_search_results.csv", "w")
csv_writer = csv.writer(file, delimiter=",")

csv_writer.writerow(["hidden weight mean", "hidden weight sd", "output weight mean", "output weight sd", "accuracy"])

#print(len(combinations))

                
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
    
    