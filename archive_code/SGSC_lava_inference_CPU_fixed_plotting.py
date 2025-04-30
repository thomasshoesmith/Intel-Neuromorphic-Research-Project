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
import csv

params = {}
params["DT_MS"] = 1.0
params["TAU_MEM"] = 20.0
params["TAU_SYN"] = 2.0
params["num_samples"] = 1 #int(11005 / 8)
params["sample_id"] = 0     #sample used for graph generation (starting at 0, < num_samples)

params["NUM_INPUT"] = 80
params["NUM_HIDDEN"] = 512
params["NUM_OUTPUT"] = 35

# toggle to record spikes, useful for debugging, but memory intensive
params["record_network_ih_activity"] =  True

# Kaggle dataset directory
dataset = 'https://www.kaggle.com/datasets/thomasshoesmith/spiking-google-speech-commands/data'

# Using opendatasets to download SGSC dataset
od.download(dataset)

x_train, y_train, x_test, y_test, x_validation, y_validation = SGSC_Loader(dir = os.getcwd() + "/spiking-google-speech-commands/",
                                                                           num_samples=params["num_samples"],
                                                                           shuffle = False,
                                                                           shuffle_seed = 42)

the_x = x_test
the_y = y_test

# transform some parmeters
decay_scale = 331

tau_mem_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_MEM"])
tau_mem_fac_int = int(np.round(tau_mem_fac*(decay_scale)))

tau_syn_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_SYN"])
tau_syn_fac_int = int(np.round(tau_syn_fac*(decay_scale)))

weight_scale = (params["TAU_SYN"] / params["DT_MS"]) * tau_syn_fac

# load connections
w_i2h = np.load("SGSC_pretrained_weights/SGSC_Pop0_Pop1-g.npy")
w_i2h = w_i2h.reshape((80,512)).T
w_i2h *= weight_scale
w_i2h *= tau_mem_fac

w_h2h = np.load("SGSC_pretrained_weights/SGSC_Pop1_Pop1-g.npy")
w_h2h = w_h2h.reshape((512,512)).T
w_h2h *= weight_scale
w_h2h *= tau_mem_fac

w_h2o = np.load("SGSC_pretrained_weights/SGSC_Pop1_Pop2-g.npy")
w_h2o = w_h2o.reshape((512,35)).T
w_h2o *= weight_scale
w_h2o *= tau_mem_fac

weight_bits= 8
max_weight = (np.max(np.concatenate((w_i2h.flatten(), w_h2h.flatten(), w_h2o.flatten()))))
min_weight = (np.min(np.concatenate((w_i2h.flatten(), w_h2h.flatten(), w_h2o.flatten()))))

w_2h_fac = 331

w_i2h_int = np.round(w_i2h*w_2h_fac).astype(np.int8)
print(f"i2h: mn == {np.amin(w_i2h_int)}, mx == {np.amax(w_i2h_int)}")

w_h2h_int = np.round(w_h2h*w_2h_fac).astype(np.int8)
print(f"h2h: mn == {np.amin(w_h2h_int)}, mx == {np.amax(w_h2h_int)}")

w_h2o_int = np.round(w_h2o*w_2h_fac).astype(np.int8)
print(f"h2o: mn == {np.amin(w_h2o_int)}, mx == {np.amax(w_h2o_int)}")

vth_hid = w_2h_fac
vth_hid_int = int(np.round(vth_hid))

sample_image_start = the_x.shape[2] * params["sample_id"]
sample_image_end = (the_x.shape[2] * params["sample_id"]) + the_x.shape[2]

the_x= np.hstack(the_x)

def run(scaling):
    # transform some parmeters
    tau_mem_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_MEM"])
    tau_mem_fac_int = int(np.round(tau_mem_fac*(scaling)))

    tau_syn_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_SYN"])
    tau_syn_fac_int = int(np.round(tau_syn_fac*(scaling)))

    input = RingBuffer(data = the_x)

    hidden = LIFReset(shape=(512, ),                         # Number and topological layout of units in the process
                    vth= vth_hid_int,                             # Membrane threshold
                    dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                    du=tau_syn_fac_int + 2,                              # Inverse synaptic time-constant
                    bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                    name="hidden",
                    reset_interval=2000)

    output = LIFReset(shape=(35, ),                         # Number and topological layout of units in the process
                    vth=2**17,                             # Membrane threshold set so it cannot spike
                    dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                    du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                    bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                    name="output",
                    reset_interval=2000)

    in_to_hid = Dense(weights= w_i2h_int,     # Initial value of the weights, chosen randomly
                name='in_to_hid')

    hid_to_hid = Dense(weights=w_h2h_int,
                    name='hid_to_hid')

    hid_to_out = Dense(weights=w_h2o_int,
                    name= 'hid_to_out')

    input.s_out.connect(in_to_hid.s_in)
    in_to_hid.a_out.connect(hidden.a_in)
    hidden.s_out.connect(hid_to_hid.s_in)
    hidden.s_out.connect(hid_to_out.s_in)
    hid_to_hid.a_out.connect(hidden.a_in)
    hid_to_out.a_out.connect(output.a_in)

    if params["record_network_ih_activity"]:
        # monitor outputs
        monitor_hidden_v = Monitor()
        monitor_hidden_v.probe(hidden.v, the_x.shape[1])

    monitor_output = Monitor()
    monitor_output.probe(output.v, the_x.shape[1])

    num_steps = int(2000/params["DT_MS"])
    # run something
    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi2SimCfg(select_tag="fixed_pt") # changed 1 -> 2

    for i in (range(the_x.shape[1] // 2000)):
        output.run(condition=run_condition, run_cfg=run_cfg)

    hidden_voltage = monitor_hidden_v.get_data()

    process = list(hidden_voltage.keys())[0]
    spikes_out = list(hidden_voltage[process].keys())[0]
    hidden_v = hidden_voltage[process][spikes_out]

    single_image = hidden_v[sample_image_start:sample_image_end]
    for i in range(params["NUM_HIDDEN"]):
        if i == 299:
            x = single_image[:,i] / 64
            np.save(f"display/{scaling}.npy", x)

    output.stop()


for i in tqdm(range(1500, 3500, 1)):
    run(i)