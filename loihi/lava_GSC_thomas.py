import numpy as np
from lava.proc.lif.process import LIFReset
from lava.proc.io.source import RingBuffer
from lava.proc.dense.process import Dense
from lava.proc.monitor.process import Monitor
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2SimCfg
from matplotlib import pyplot as plt
from tqdm import tqdm
import typing as ty

import gsc_dataset_loader

# Import Process level primitives
from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.process.variable import Var
from lava.magma.core.process.ports.ports import InPort, OutPort

# Import parent classes for ProcessModels
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.model.py.model import PyLoihiProcessModel

# Import ProcessModel ports, data-types
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

# Import execution protocol and hardware resources
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU

# Import decorators
from lava.magma.core.decorator import implements, requires

from lava.proc.spike_input.process import SpikeInput

params = {}
params["DT_MS"] = 1.0
params["TAU_MEM"] = 20.0
params["TAU_SYN"] = 5.0
params["num_samples"] = 10
params["sample_id"] = 0

params["NUM_INPUT"] = 80
params["NUM_HIDDEN"] = 512
params["NUM_OUTPUT"] = 35
params["TRIAL_STEPS"] = 1000

# load gsc dataset
train_x, train_y, validation_x, validation_y, test_x, test_y = gsc_dataset_loader.load_gsc("/Users/tn41/data/rawSC/rawSC_80input/", 
                                                                                           1, 
                                                                                           80,
                                                                                           params.get("num_samples"),
                                                                                           True, shuffle= False)

the_x = test_x
the_y = test_y

# transform some parmeters
tau_mem_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_MEM"])
print(tau_mem_fac)
tau_syn_fac = 1.0-np.exp(-params["DT_MS"]/params["TAU_SYN"])
print(tau_syn_fac)

# load connections
w_i2h = np.load("99-Conn_Pop0_Pop1-g.npy")
w_i2h = w_i2h.reshape((80,512)).T
w_i2h *= tau_mem_fac
#w_i2h /= p["TAU_MEM"]
w_h2h = np.load("99-Conn_Pop1_Pop1-g.npy")
w_h2h = w_h2h.reshape((512,512)).T
w_h2h *= tau_mem_fac
#w_h2h /= p["TAU_MEM"]
w_h2o = np.load("99-Conn_Pop1_Pop2-g.npy")
w_h2o = w_h2o.reshape((512,35)).T
w_h2o *= tau_mem_fac
#w_h2o /= p["TAU_MEM"]


sample_image_start = int(the_x.shape[0] / params["num_samples"] * params["sample_id"])
sample_image_end = int((the_x.shape[0] / params["num_samples"] * params["sample_id"]) + the_x.shape[0] / params["num_samples"])

input = SpikeInput(images=the_x,
                   vth=1.,
                   dt_ms=params["DT_MS"],
                   tau_mem=params["TAU_MEM"],
                   num_steps_per_image=params["TRIAL_STEPS"])

hidden = LIFReset(shape=(512, ),                         # Number and topological layout of units in the process
                  vth=1.,                             # Membrane threshold
                  dv=tau_mem_fac,                              # Inverse membrane time-constant
                  du=tau_syn_fac,                              # Inverse synaptic time-constant
                  bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                  name="hidden",
                  reset_interval=params["TRIAL_STEPS"])

output = LIFReset(shape=(35, ),                         # Number and topological layout of units in the process
                  vth=1e9,                             # Membrane threshold set so it cannot spike
                  dv=tau_mem_fac,                              # Inverse membrane time-constant
                  du=tau_syn_fac,                              # Inverse synaptic time-constant
                  bias_mant=0.0,           # Bias added to the membrane voltage in every timestep
                  name="output",
                  reset_interval=params["TRIAL_STEPS"])

in_to_hid = Dense(weights= w_i2h,     # Initial value of the weights, chosen randomly
                  name='in_to_hid')

hid_to_hid = Dense(weights=w_h2h,
                   name='hid_to_hid')

hid_to_out = Dense(weights=w_h2o,
                   name= 'hid_to_out')

input.spikes_out.connect(in_to_hid.s_in)
in_to_hid.a_out.connect(hidden.a_in)
hidden.s_out.connect(hid_to_hid.s_in)
hidden.s_out.connect(hid_to_out.s_in)
hid_to_hid.a_out.connect(hidden.a_in)
hid_to_out.a_out.connect(output.a_in)

# monitor outputs
monitor_input = Monitor()
monitor_hidden = Monitor()
monitor_output = Monitor()
    
num_steps = int(1000/params["DT_MS"])

print("number of samples:", params["num_samples"])

monitor_input.probe(input.spikes_out, the_x.shape[0])
monitor_hidden.probe(hidden.s_out, the_x.shape[0])
monitor_output.probe(output.v, the_x.shape[0])

# run something
run_condition = RunSteps(num_steps=num_steps)
run_cfg = Loihi2SimCfg(select_tag="floating_pt") # changed 1 -> 2

n_sample = params.get("num_samples")


for i in tqdm(range(params.get("num_samples"))):
    output.run(condition=run_condition, run_cfg=run_cfg)
    
output_v = monitor_output.get_data()
good = 0
    
for i in range(params["num_samples"]):
    out_v = output_v["output"]["v"][i*num_steps:(i+1)*num_steps,:]
    sum_v = np.sum(out_v,axis=0)
    pred = np.argmax(sum_v)
    print(f"Pred: {pred}, True:{the_y[i]}")
    if pred == the_y[i]:
        good += 1
            
print(f"test accuracy: {good/n_sample*100}")
#output.stop()


# Input spike activity

input_spikes = monitor_input.get_data()
process = list(input_spikes.keys())[0]
spikes_out = list(input_spikes[process].keys())[0]
input_s = input_spikes[process][spikes_out]

input_single_image = input_s[sample_image_start:sample_image_end]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Input Spikes (with raw input for comparison) from lava')

for i in range(params["NUM_INPUT"]):
    ax1.scatter(np.where(input_single_image[:,i] > 0)[0], 
                np.where(input_single_image[:,i] > 0)[0].shape[0] * [i],
                c = 'b',
                s = 2)
    
# no clue why I need to rotate and flip...could be investigated? 
ax2.imshow(np.flipud(np.rot90(the_x[sample_image_start:sample_image_end], 1)), aspect = 'auto')

ax1.set_ylim(0, 80)
ax2.set_ylim(0, 80)
ax1.set_xlim(0, the_x.shape[0] / params["num_samples"])
ax2.set_xlim(0, the_x.shape[0] / params["num_samples"])

fig.tight_layout()

    
# Hidden layer activity 

hidden_spikes = monitor_hidden.get_data()

process = list(hidden_spikes.keys())[0]
spikes_out = list(hidden_spikes[process].keys())[0]
hidden_s = hidden_spikes[process][spikes_out]

hidden_single_image = hidden_s[sample_image_start:sample_image_end]

plt.figure()
for i in range(params["NUM_HIDDEN"]):
    plt.scatter(np.where(hidden_single_image[:,i] > 0)[0], 
                np.where(hidden_single_image[:,i] > 0)[0].shape[0] * [i],
                c = 'b',
                s = 0.5)
    
plt.title("Hidden layer spiking activity")
plt.ylabel("layer")
plt.xlabel("timesteps")
plt.xlim([0,params["TRIAL_STEPS"]])

# output voltage activity
# high voltage levels are explained by a mega high threshold, to enable non-spiking
output_voltage = monitor_output.get_data()
    
process = list(output_voltage.keys())[0]
spikes_out = list(output_voltage[process].keys())[0]
output_v = output_voltage[process][spikes_out]

single_image = output_v[sample_image_start:sample_image_end]

plt.figure()
for i in range(params["NUM_OUTPUT"]):
    plt.plot(single_image[:,i])
    
plt.title("output layer voltage activity")
plt.ylabel("voltage (v)")
plt.xlabel("timesteps")
plt.xlim([0,params["TRIAL_STEPS"]])
plt.show()

output.stop()
