import os 
import sys
import json
import re
from HD_eventprop import hd_eventprop
from tqdm import trange
from numba import cuda

# get list of json param files
param_dir = "jade_test_params_04"
dir_list = os.listdir(param_dir)
dir_cwd = os.getcwd()

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

dir_list = natural_sort(dir_list)
dir_list = [(param_dir + "/") + s for s in dir_list]

for i in trange(len(dir_list)):
    os.chdir(dir_cwd)
    #reset gpu memory
    device = cuda.get_current_device()
    device.reset()
    with open(dir_list[i], "r") as f:
        params = json.load(f)

    os.chdir("output")
    # nested directory for sweep
    if len(params.get("sweeping_suffix")) > 0:
        # change dir for readout files
        try:
            os.makedirs(params.get("output_dir"))
        except:
            pass

        os.chdir(params.get("output_dir"))

    accuracy = hd_eventprop(params, 
                            file_path = os.path.expanduser("~/data/GSC/experimental_1/"))
