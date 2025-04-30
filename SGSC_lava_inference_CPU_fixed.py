import numpy as np
from lava.proc.lif.process import LIFReset
from lava.proc.io.source import RingBuffer
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi2SimCfg
from tqdm import trange
import os
import opendatasets as od
from SGSC_dataset_loader_padded_spikes import SGSC_Loader

def rescale_factor(w, bits, percentile = 100):
    rng = float(2**(bits-1))
    mx = max(np.percentile(w,percentile), np.percentile(-w,percentile))
    fac = (rng-1)/mx
    mn = -(rng-2)/fac
    return (fac, mn, mx)

params = {}
params["DT_MS"] = 1.0
params["TAU_MEM"] = 20.0
params["TAU_SYN"] = 5.0
params["num_samples"] = 10 #11005
params["sample_id"] = 0     #sample used for graph generation (starting at 0, < num_samples)
params["return_single_sample"] = None

params["NUM_INPUT"] = 80
params["NUM_HIDDEN"] = 512
params["NUM_OUTPUT"] = 35

params["recurrent"] = True
params["weights_dir"] = "SGSC_pretrained_weights_recurrent"
params["bit"] = 8
params["timesteps"] = 2000

# toggle to record spikes, useful for debugging, but memory intensive
params["record_network_ih_activity"] =  False

# Kaggle dataset directory
dataset = 'https://www.kaggle.com/datasets/thomasshoesmith/spiking-google-speech-commands/data'

# Using opendatasets to download SGSC dataset
od.download(dataset)

x_train, y_train, x_test, y_test, x_validation, y_validation = SGSC_Loader(dir = os.getcwd() + "/spiking-google-speech-commands/",
                                                                           num_samples=params["num_samples"],
                                                                           shuffle = True,
                                                                           shuffle_seed = 0,
                                                                           return_single_sample = params["return_single_sample"])

the_x = x_test
the_y = y_test

tau_mem_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_MEM"])
tau_mem_fac_int = int(np.round(tau_mem_fac*(2**12)))

tau_syn_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_SYN"])
tau_syn_fac_int = int(np.round(tau_syn_fac*(2**12)))

weight_scale = 1 #(params["TAU_SYN"] / params["DT_MS"]) * tau_syn_fac

# load connections
w_i2h = np.load(f"{params['weights_dir']}/SGSC_Pop0_Pop1-g.npy")
w_i2h = w_i2h.reshape((params["NUM_INPUT"],
                       params["NUM_HIDDEN"])).T
w_i2h *= weight_scale
w_i2h *= tau_mem_fac

if params["recurrent"]:
    w_h2h = np.load(f"{params['weights_dir']}/SGSC_Pop1_Pop1-g.npy")
    w_h2h = w_h2h.reshape((params["NUM_HIDDEN"],
                           params["NUM_HIDDEN"])).T
    w_h2h *= weight_scale
    w_h2h *= tau_mem_fac

w_h2o = np.load(f"{params['weights_dir']}/SGSC_Pop1_Pop2-g.npy")
w_h2o = w_h2o.reshape((params["NUM_HIDDEN"],
                       params["NUM_OUTPUT"])).T
w_h2o *= weight_scale
w_h2o *= tau_mem_fac

print(f"i2h: mn == {np.amin(w_i2h)}, mx == {np.amax(w_i2h)}")
if params["recurrent"]: print(f"h2h: mn == {np.amin(w_h2h)}, mx == {np.amax(w_h2h)}")
print(f"h2o: mn == {np.amin(w_h2o)}, mx == {np.amax(w_h2o)}")

max_weight = (np.max(np.concatenate((w_i2h.flatten(), w_h2h.flatten() if params["recurrent"] else np.array([]), w_h2o.flatten()))))
min_weight = (np.min(np.concatenate((w_i2h.flatten(), w_h2h.flatten() if params["recurrent"] else np.array([]), w_h2o.flatten()))))

bit_range = float(2**(params["bit"]-1)) - 1
weight_scale = bit_range/(max(abs(max_weight), abs(min_weight)))

w_i2h_int = np.round(w_i2h*weight_scale).astype(np.int8)
print(f"i2h: mn == {np.amin(w_i2h_int)}, mx == {np.amax(w_i2h_int)}")

if params["recurrent"]:
    w_h2h_int = np.round(w_h2h*weight_scale).astype(np.int8)
    print(f"h2h: mn == {np.amin(w_h2h_int)}, mx == {np.amax(w_h2h_int)}")

w_h2o_int = np.round(w_h2o*weight_scale).astype(np.int8)
print(f"h2o: mn == {np.amin(w_h2o_int)}, mx == {np.amax(w_h2o_int)}")

vth_hid_int = int(np.round(weight_scale))

sample_image_start = the_x.shape[2] * params["sample_id"]
sample_image_end = (the_x.shape[2] * params["sample_id"]) + the_x.shape[2]

save_weights = False
if save_weights:
    np.save("SGSC_pretrained_weights_recurrent_quantised/SGSC_Pop0_Pop1-g.npy", np.reshape(w_i2h_int, params["NUM_INPUT"] * params["NUM_HIDDEN"]))
    np.save("SGSC_pretrained_weights_recurrent_quantised/SGSC_Pop1_Pop1-g.npy", np.reshape(w_h2h_int, params["NUM_HIDDEN"] * params["NUM_HIDDEN"]))
    np.save("SGSC_pretrained_weights_recurrent_quantised/SGSC_Pop1_Pop2-g.npy", np.reshape(w_h2o_int, params["NUM_HIDDEN"] * params["NUM_OUTPUT"]))
    exit()

the_x= np.hstack(the_x)
print(the_x.shape)

input = RingBuffer(data = the_x)

hidden = LIFReset(shape=(params["NUM_HIDDEN"], ),                         # Number and topological layout of units in the process
                  vth= vth_hid_int,                             # Membrane threshold
                  dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                  du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                  bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                  name="hidden",
                  reset_interval=params["timesteps"])

output = LIFReset(shape=(params["NUM_OUTPUT"], ),                         # Number and topological layout of units in the process
                  vth=2**17,                             # Membrane threshold set so it cannot spike
                  dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                  du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                  bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                  name="output",
                  reset_interval=params["timesteps"])

in_to_hid = Dense(weights= w_i2h_int,     # Initial value of the weights, chosen randomly
              name='in_to_hid')
if params["recurrent"]:
    hid_to_hid = Dense(weights=w_h2h_int,
                    name='hid_to_hid')

hid_to_out = Dense(weights=w_h2o_int,
                   name= 'hid_to_out')

input.s_out.connect(in_to_hid.s_in)
in_to_hid.a_out.connect(hidden.a_in)
if params["recurrent"]: hidden.s_out.connect(hid_to_hid.s_in)
hidden.s_out.connect(hid_to_out.s_in)
if params["recurrent"]: hid_to_hid.a_out.connect(hidden.a_in)
hid_to_out.a_out.connect(output.a_in)

if params["record_network_ih_activity"]:
    # monitor outputs
    monitor_input = Monitor()
    monitor_hidden = Monitor()
    monitor_hidden_v = Monitor()

    monitor_hidden_v.probe(hidden.v, the_x.shape[1])

    monitor_input.probe(input.s_out, the_x.shape[1])
    monitor_hidden.probe(hidden.s_out, the_x.shape[1])

monitor_output = Monitor()
monitor_output.probe(output.v, the_x.shape[1])

num_steps = int(params["timesteps"]/params["DT_MS"])
print("number of samples:", params["num_samples"])
if params["return_single_sample"] != None: print("! overridden, selected sample run !")

# run something
run_condition = RunSteps(num_steps=num_steps)
run_cfg = Loihi2SimCfg(select_tag="fixed_pt") # changed 1 -> 2

n_sample = params.get("num_samples")

for i in trange(the_x.shape[1] // params["timesteps"]):
    output.run(condition=run_condition, run_cfg=run_cfg)

output_v = monitor_output.get_data()
good = 0

for i in range(the_x.shape[1] // params["timesteps"]):
    out_v = output_v["output"]["v"][i*num_steps:(i+1)*num_steps,:]
    sum_v = np.sum(out_v,axis=0)
    pred = np.argmax(sum_v)
    print(f"prediction {pred} vs ground truth {the_y[i]}")
    if pred == the_y[i]:
        good += 1

print(f"test accuracy: {good/len(the_y)*100}")
output.stop()