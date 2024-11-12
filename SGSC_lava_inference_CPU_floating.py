import numpy as np
from lava.proc.lif.process import LIFReset
from lava.proc.io.source import RingBuffer
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2SimCfg
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import opendatasets as od
from SGSC_dataset_loader_padded_spikes import SGSC_Loader

params = {}
params["DT_MS"] = 1.0
params["TAU_MEM"] = 20.0
params["TAU_SYN"] = 2.0
params["num_samples"] = 100 #11005
params["sample_id"] = 0     #sample used for graph generation (starting at 0, < num_samples)

params["NUM_INPUT"] = 80
params["NUM_HIDDEN"] = 512
params["NUM_OUTPUT"] = 35

params["timesteps"] = 2000

params["recurrent"] = False
params["weights_dir"] = "SGSC_pretrained_weights_4"

# toggle to record spikes, useful for debugging, but memory intensive
params["record_network_ih_activity"] =  False

# Kaggle dataset directory
dataset = 'https://www.kaggle.com/datasets/thomasshoesmith/spiking-google-speech-commands/data'

# Using opendatasets to download SGSC dataset
od.download(dataset)

x_train, y_train, x_test, y_test, x_validation, y_validation = SGSC_Loader(dir = os.getcwd() + "/data/", #/spiking-google-speech-commands/",
                                                                           num_samples=params["num_samples"],
                                                                           shuffle = True,
                                                                           shuffle_seed = 0)

the_x = x_test
the_y = y_test

# transform some parmeters
tau_mem_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_MEM"])
tau_syn_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_SYN"])
weight_scale = (params["TAU_SYN"] / params["DT_MS"]) * tau_syn_fac

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

sample_image_start = the_x.shape[2] * params["sample_id"]
sample_image_end = (the_x.shape[2] * params["sample_id"]) + the_x.shape[2]

the_x= np.hstack(the_x)
print(the_x.shape)

input = RingBuffer(data = the_x)

hidden = LIFReset(shape=(params["NUM_HIDDEN"], ),     # Number and topological layout of units in the process
                  vth=1.,                             # Membrane threshold
                  dv=tau_mem_fac,                              # Inverse membrane time-constant
                  du=tau_syn_fac,                              # Inverse synaptic time-constant
                  bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                  name="hidden",
                  reset_interval=params["timesteps"])

output = LIFReset(shape=(params["NUM_OUTPUT"], ),                         # Number and topological layout of units in the process
                  vth=2**17,                             # Membrane threshold set so it cannot spike
                  dv=tau_mem_fac,                              # Inverse membrane time-constant
                  du=tau_syn_fac,                              # Inverse synaptic time-constant
                  bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                  name="output",
                  reset_interval=params["timesteps"])

in_to_hid = Dense(weights= w_i2h,     # Initial value of the weights, chosen randomly
              name='in_to_hid')

if params["recurrent"]:
    hid_to_hid = Dense(weights=w_h2h,
                    name='hid_to_hid')

hid_to_out = Dense(weights=w_h2o,
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

# run something
run_condition = RunSteps(num_steps=num_steps)
run_cfg = Loihi2SimCfg(select_tag="floating_pt") # changed 1 -> 2

n_sample = params.get("num_samples")

for i in tqdm(range(the_x.shape[1] // params["timesteps"])):
    output.run(condition=run_condition, run_cfg=run_cfg)

output_v = monitor_output.get_data()
good = 0

for i in range(the_x.shape[1] // params["timesteps"]):
    out_v = output_v["output"]["v"][i*num_steps:(i+1)*num_steps,:]
    sum_v = np.sum(out_v,axis=0)
    pred = np.argmax(sum_v)
    print(f"Pred: {pred}, True:{the_y[i]}")
    if pred == the_y[i]:
        good += 1

print(f"test accuracy: {good/n_sample*100}")
output.stop()

if params["record_network_ih_activity"]:
    # Input spike activity
    input_spikes = monitor_input.get_data()

    process = list(input_spikes.keys())[0]
    spikes_out = list(input_spikes[process].keys())[0]
    input_s = input_spikes[process][spikes_out]

    input_single_image = input_s[sample_image_start:sample_image_end]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Input Spikes (with past input for comparison) from lava')

    for i in range(params["NUM_INPUT"]):
        ax1.scatter(np.where(input_single_image[:,i] > 0)[0], 
                    np.where(input_single_image[:,i] > 0)[0].shape[0] * [i],
                    c = '#1f77b4',
                    s = 2)

    x = np.swapaxes(the_x, 0, 1)[sample_image_start:sample_image_end]
    for i in range(params["NUM_INPUT"]):
        ax2.scatter(np.where(x[:,i] > 0)[0], 
                    np.where(x[:,i] > 0)[0].shape[0] * [i],
                    c = '#1f77b4',
                    s = 2)

    ax1.set_ylim(0, 80)
    ax2.set_ylim(0, 80)
    ax1.set_xlim(0, the_x.shape[1] / params["num_samples"])
    ax2.set_xlim(0, the_x.shape[1] / params["num_samples"])

    fig.tight_layout()

    plt.show()
    
if params["record_network_ih_activity"]:
    # Hidden layer activity 

    hidden_spikes = monitor_hidden.get_data()

    process = list(hidden_spikes.keys())[0]
    spikes_out = list(hidden_spikes[process].keys())[0]
    hidden_s = hidden_spikes[process][spikes_out]

    hidden_single_image = hidden_s[sample_image_start:sample_image_end]

    for i in range(params["NUM_HIDDEN"]):
        plt.scatter(np.where(hidden_single_image[:,i] > 0)[0], 
                    np.where(hidden_single_image[:,i] > 0)[0].shape[0] * [i],
                    c = '#1f77b4',
                    s = 0.5)

    plt.title("Hidden layer spiking activity")
    plt.ylim(0, params["NUM_HIDDEN"])
    plt.xlim(0, the_x.shape[1] / params["num_samples"])
    plt.ylabel("layer")
    plt.xlabel("timesteps")
    plt.show()
    
if params["record_network_ih_activity"]:
    # hidden voltage activity
    hidden_voltage = monitor_hidden_v.get_data()

    process = list(hidden_voltage.keys())[0]
    spikes_out = list(hidden_voltage[process].keys())[0]
    hidden_v = hidden_voltage[process][spikes_out]

    single_image = hidden_v[sample_image_start:sample_image_end]
    plt.figure(figsize=(12, 3), dpi=80)
    for i in range(params["NUM_HIDDEN"]):
        if i == 299:
            plt.plot(single_image[:,i])
        
    plt.title("Hidden layer voltage activity")
    plt.ylabel("voltage (v)")
    plt.xlabel("timesteps")
    plt.xlim(0, the_x.shape[1] / params["num_samples"])
    #plt.xlim(450, 600)
    """
    plt.xlim(490, 520)
    plt.ylim(0, 1.0)

    for i in range(1000):
        plt.axvline(i * 2, color = "red", alpha=0.5, linestyle = "dashed", label = f"timestep {256 * 4}")
    plt.axhline(1 * 0.25, color = "red", alpha=0.5, linestyle = "dashed")
    plt.axhline(1 * 0.50, color = "red", alpha=0.5, linestyle = "dashed")
    plt.axhline(1 * 0.75, color = "red", alpha=0.5, linestyle = "dashed")
    """
    plt.show()
    
if params["record_network_ih_activity"]:
    # hidden voltage activity
    hidden_voltage = monitor_hidden_v.get_data()

    process = list(hidden_voltage.keys())[0]
    spikes_out = list(hidden_voltage[process].keys())[0]
    hidden_v = hidden_voltage[process][spikes_out]

    single_image = hidden_v[sample_image_start:sample_image_end]
    plt.figure(figsize=(12, 3), dpi=80)
    for i in range(params["NUM_HIDDEN"]):
        if i == 299:
            plt.plot(single_image[:,i]) # / (2 ** weight_bits / 2))
    
    plt.scatter(np.where(hidden_single_image[:,299] > 0)[0], 
                np.where(hidden_single_image[:,299] > 0)[0].shape[0] * [1.0],
                c = "r",
                label = "spikes")
        
    plt.title("Hidden layer voltage activity (floating)")
    plt.ylabel("voltage (v)")
    plt.xlabel("timesteps")
    plt.xlim(0, the_x.shape[1] / params["num_samples"])
    #plt.xlim(1600, 1900)
    plt.xlim(450, 600)
    plt.ylim(0, 1 * 1.1)

    for i in range(1000):
        plt.axvline(i * 10, color = "grey", alpha=0.5, linestyle = "dashed")

    plt.axhline(1.0, color = "red", alpha=0.5, linestyle = "dashed", label = "threshold")

    plt.legend()
    plt.show()
    
if params["record_network_ih_activity"]:
    # output voltage activity
    # high voltage levels are explained by a mega high threshold, to enable non-spiking
    output_voltage = monitor_output.get_data()

    process = list(output_voltage.keys())[0]
    spikes_out = list(output_voltage[process].keys())[0]
    output_v = output_voltage[process][spikes_out]

    single_image = output_v[sample_image_start:sample_image_end]
    plt.figure(figsize=(12, 3), dpi=80)
    for i in range(params["NUM_OUTPUT"]):
        plt.plot(single_image[:,i])
        
    plt.title("Output layer voltage activity")
    plt.ylabel("voltage (v)")
    plt.xlabel("timesteps")
    plt.xlim(0, the_x.shape[1] / params["num_samples"])
    plt.show()
    
if params["record_network_ih_activity"]:
    # output voltage activity
    # high voltage levels are explained by a mega high threshold, to enable non-spiking
    output_voltage = monitor_output.get_data()

    process = list(output_voltage.keys())[0]
    spikes_out = list(output_voltage[process].keys())[0]
    output_v = output_voltage[process][spikes_out]

    single_image = output_v[sample_image_start:sample_image_end]
    plt.figure(figsize=(12, 3), dpi=80)
    for i in range(params["NUM_OUTPUT"]):
        if i == 16:
            plt.plot(single_image[:,i])
        
    plt.title("Output layer voltage activity")
    plt.ylabel("voltage (v)")
    plt.xlabel("timesteps")
    plt.xlim(0, the_x.shape[1] / params["num_samples"])
    plt.show()