import numpy as np
from lava.proc.lif.process import LIFReset
from lava.proc.io.source import RingBuffer
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg
from lava.magma.core.run_configs import Loihi2HwCfg
from matplotlib import pyplot as plt
from tqdm import tqdm
from lava.magma.core.process.process import LogConfig
import logging
from lava.utils.system import Loihi2
import os
from lava.proc.embedded_io.spike import PyToNxAdapter
from lava.proc.embedded_io.state import Read as StateReader
from lava.proc.io.sink import RingBuffer as SpikeOut
from lava.proc.io.sink import PyReceiveModelFixed
from lava.utils.loihi2_state_probes import StateProbe

import opendatasets as od
from SGSC_dataset_loader_padded_spikes import SGSC_Loader

def rescale_factor(w, bits):
    rng = float(2**(bits-1))
    mx = max(np.percentile(w,99), np.percentile(-w,99))
    fac = (rng-1)/mx
    mn = -(rng-2)/fac
    return (fac, mn, mx)

log_config = LogConfig("lava_SGSC.log")
log_config.level_console= logging.WARNING

params = {}
params["DT_MS"] = 1.0
params["TAU_MEM"] = 20.0
params["TAU_SYN"] = 2.0
params["num_samples"] = 1 #int(11005 / 8)
params["sample_id"] = 0     #sample used for graph generation (starting at 0, < num_samples)

params["NUM_INPUT"] = 80
params["NUM_HIDDEN"] = 512
params["NUM_OUTPUT"] = 35

params["timesteps"] = 2048

# toggle to record spikes, useful for debugging, but memory intensive
params["record_network_ih_activity"] =  False

os.environ["PATH"] += ":/nfs/ncl/bin:"
os.environ["PARTITION"] = "oheogulch_20m" # _2h (if 2 hours are needed)
os.environ['SLURM'] = '1'
os.environ['LOIHI_GEN'] = 'N3C1'

#os.environ['LOIHI_GEN'] = 'N3C1'
#os.environ['NOSLURM'] = '1'
#os.environ['NXSDKHOST'] = '10.1.23.175'
#os.environ['HOST_BINARY'] = '/opt/nxcore/bin/nx_driver_server'
#os.environ["PATH"] += ":/opt/riscv/bin/:"

loihi2_is_available = Loihi2.is_loihi2_available


if loihi2_is_available:
    print(f'Running on {Loihi2.partition}')
    from lava.utils import loihi2_profiler
else:
    RuntimeError("Loihi2 compiler is not available in this system. "
                 "This tutorial cannot proceed further.")

do_plots= True
weight_bits= 8

# Kaggle dataset directory
dataset = 'https://www.kaggle.com/datasets/thomasshoesmith/spiking-google-speech-commands/data'

# Using opendatasets to download SGSC dataset
od.download(dataset)

x_train, y_train, x_test, y_test, x_validation, y_validation = SGSC_Loader(dir = os.getcwd() + "/spiking-google-speech-commands/",
                                                                           num_samples=params["num_samples"],
                                                                           shuffle = False,
                                                                           number_of_timesteps = params["timesteps"])

the_x = x_test
the_y = y_test

print(the_x[0].shape)

weight_bits= 8

# transform some parmeters
tau_mem_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_MEM"])
tau_mem_fac_int = int(np.round(tau_mem_fac*(2**12)))
tau_syn_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_SYN"])
tau_syn_fac_int = int(np.round(tau_syn_fac*(2**12)))

# load connections
w_i2h = np.load("SGSC_pretrained_weights/SGSC_Pop0_Pop1-g.npy")
w_i2h = w_i2h.reshape((params["NUM_INPUT"],
                       params["NUM_HIDDEN"])).T
w_i2h *= tau_mem_fac

w_h2h = np.load("SGSC_pretrained_weights/SGSC_Pop1_Pop1-g.npy")
w_h2h = w_h2h.reshape((params["NUM_HIDDEN"],
                       params["NUM_HIDDEN"])).T
w_h2h *= tau_mem_fac

w_h2o = np.load("SGSC_pretrained_weights/SGSC_Pop1_Pop2-g.npy")
w_h2o = w_h2o.reshape((params["NUM_HIDDEN"],
                       params["NUM_OUTPUT"])).T
w_h2o *= tau_mem_fac

# weight scaling
w = np.hstack([w_i2h,w_h2h])
w_2h_fac, mn, mx = rescale_factor(w,weight_bits)
w_i2h[w_i2h > mx] = mx
w_i2h[w_i2h < mn] = mn
w_i2h_int = np.round(w_i2h*w_2h_fac).astype(np.int8)
print(f"i2h: mn == {np.amin(w_i2h_int)}, mx == {np.amax(w_i2h_int)}")

w_h2h[w_h2h > mx] = mx
w_h2h[w_h2h < mn] = mn
w_h2h_int = np.round(w_h2h*w_2h_fac).astype(np.int8)
print(f"h2h: mn == {np.amin(w_h2h_int)}, mx == {np.amax(w_h2h_int)}")

w_2o_fac, mn, mx = rescale_factor(w_h2o,weight_bits)
w_2o_fac /= 2.0
mn *= 2.0
mx *= 2.0
w_h2o[w_h2o > mx] = mx
w_h2o[w_h2o < mn] = mn
w_h2o_int = np.round(w_h2o*w_2o_fac).astype(np.int8)
print(f"h2o: mn == {np.amin(w_h2o_int)}, mx == {np.amax(w_h2o_int)}")

vth_hid = w_2h_fac
vth_hid_int = int(np.round(vth_hid))

the_x= np.hstack(the_x)
print(the_x.shape)

# Create processes
input = RingBuffer(data=the_x)

py2nx_inp = PyToNxAdapter(shape=(the_x.shape[0],))

hidden = LIFReset(shape=(params["NUM_HIDDEN"], ),                         # Number and topological layout of units in the process
                  vth=vth_hid_int,                             # Membrane threshold
                  dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                  du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                  bias_mant=0,           # Bias added to the membrane voltage in every timestep
                  name="hidden",
                  reset_interval=params["timesteps"],
                  log_config=log_config)

output = LIFReset(shape=(params["NUM_OUTPUT"], ),                         # Number and topological layout of units in the process
                  vth=2**30,                             # Membrane threshold
                  dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                  du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                  bias_mant=0,           # Bias added to the membrane voltage in every timestep
                  name="output",
                  reset_interval=params["timesteps"],
                  log_config=log_config)

in_to_hid = Dense(weights= w_i2h_int,     # Initial value of the weights, chosen randomly
                  name='in_to_hid')

#probe_w = StateProbe(in_to_hid.weights)
probe_v = StateProbe(output.v)

hid_to_hid = Dense(weights=w_h2h_int,
                   name='hid_to_hid')

hid_to_out = Dense(weights=w_h2o_int,
                   name= 'hid_to_out')

input.s_out.connect(py2nx_inp.inp)
py2nx_inp.out.connect(in_to_hid.s_in)
in_to_hid.a_out.connect(hidden.a_in)
hidden.s_out.connect(hid_to_hid.s_in)
hidden.s_out.connect(hid_to_out.s_in)
hid_to_hid.a_out.connect(hidden.a_in)
hid_to_out.a_out.connect(output.a_in)

# monitor outputs
num_steps = int(params["timesteps"]/params["DT_MS"])

# run something
run_condition = RunSteps(num_steps=num_steps)

loihi2hw_exception_map = {
            SpikeOut: PyReceiveModelFixed,
        }

run_cfg = Loihi2HwCfg(exception_proc_model_map=loihi2hw_exception_map,callback_fxs=[probe_v])
output._log_config.level = logging.INFO

for i in tqdm(range(params["num_samples"])):
    output.run(condition=run_condition, run_cfg=run_cfg)

print(probe_v.time_series[:num_steps])
output.stop()
output_v = probe_v.time_series.reshape(num_steps * params["num_samples"], params["NUM_OUTPUT"]).T
print(output_v.shape)
print(output_v)

good = 0
for i in range(params["num_samples"]):
    out_v = output_v[:,i*num_steps:(i+1)*num_steps]
    print(np.sum(output_v))
    sum_v = np.sum(out_v,axis=1)
    print(sum_v)
    print(the_y[i])
    pred = np.argmax(sum_v)
    # print(f"Pred: {pred}, True:{Y_test[i]}")
    if pred == the_y[i]:
        good += 1
    if do_plots:
        plt.figure()
        plt.plot(out_v.T)
        plt.show()

print("test accuracy: ", good/params["num_samples"]*100)