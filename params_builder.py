import json
import os
import sys
import itertools

#combinations
combinations = {}
<<<<<<< HEAD
directory_name = "rawHD_coarse_weight_sweep"
=======
directory_name = "rawSC_coarse_weight_sweep"
>>>>>>> 72b1b68a16d84efdf8f34b55091924d257f4b947
params = {}

previous_d = os.getcwd()
os.chdir(os.path.expanduser("~/PhD/Intel-Neuromorphic-Research-Project/"))
print(os.getcwd())

try:
    os.mkdir(directory_name)
    print("directory made")
except:
    pass

os.chdir(previous_d) #fix this TODO:

if len(sys.argv) != 2:
    print("running local parameters")
    with open("SC_params.json", "r") as f:
        params = json.load(f)

else:
    print("running passed arguments")
    with open(sys.argv[1], "r") as f:
        params = json.load(f)

# change the sweep parameters here
combinations["input_hidden_w_mean"] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
combinations["input_hidden_w_sd"] = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
combinations["hidden_hidden_w_mean"] = [0.0]
combinations["hidden_hidden_w_sd"] = [0.1, 0.2, 0.3, 0.4, 0.5]
combinations["hidden_output_w_mean"] = [3.0]
combinations["hidden_output_w_sd"] = [1.5]

combined = []

for c in combinations:
    combined.append(combinations.get(c))

unique_combo = list(itertools.product(*combined))

for i_count, i in enumerate(unique_combo):
    for c_count, c in enumerate(combinations):
        params[c] = i[c_count]

    params["sweeping_suffix"] = "-" + str(i_count)

    json_object = json.dumps(params, indent = 4)
    print(json_object)
    with open(f"{directory_name}/params_{i_count}.json", "w") as outfile:
        outfile.write(json_object)