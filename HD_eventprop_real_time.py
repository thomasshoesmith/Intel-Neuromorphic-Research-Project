#export CUDA_PATH=/usr/local/cuda
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from tqdm import trange

from HD_eventprop import hd_eventprop

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
params["verbose"] = True
params["debug"] = True
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

accuracy, v = hd_eventprop(params, 
                        output_dir = "HD_eventprop_rt",
                        model_description = "rt")

print(f"accuracy of the network is {accuracy * 100:.2f}%")

plt.plot(v)
plt.show()