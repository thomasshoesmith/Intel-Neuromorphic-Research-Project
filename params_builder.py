import json
import os
import sys
#combinations
combinations = {}
directory_name = "jade_test_params_03"
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
    print("error: please pass json file")
    exit()

else:
    print("running passed arguments")
    with open(sys.argv[1], "r") as f:
        params = json.load(f)

combinations["aug_swap_pixels_pSwap"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Writing to sample.json
# horrid solution TODO: improve this code to support cross combinations
len_output_dir = len(params.get("output_dir"))
for c_count, c in enumerate(combinations):
    for i_count, i in enumerate(combinations.get(c)):
        params[c] = i
        #params["output_dir"] = params.get("output_dir")[:len_output_dir] + "_" + str(i_count + (c_count * len(combinations)))
        params["sweeping_suffix"] = "-" + str(i_count + (c_count * len(combinations)))

        json_object = json.dumps(params, indent = 4)
        print(json_object)
        with open(f"{directory_name}/params_{i_count + (c_count * len(combinations))}.json", "w") as outfile:
            outfile.write(json_object)

