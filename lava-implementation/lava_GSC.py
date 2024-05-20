# script by Thomas Nowotny,
# modified by Thomas Shoesmith

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
import utils
import GSC_utils
from lava.utils.system import Loihi2
import os
from lava.proc.embedded_io.spike import PyToNxAdapter
from lava.proc.embedded_io.state import Read as StateReader
from lava.proc.io.sink import RingBuffer as SpikeOut
from lava.proc.io.sink import PyReceiveModelFixed
from lava.utils.loihi2_state_probes import StateProbe

log_config = LogConfig("lava_GSC.log")
log_config.level_console= logging.WARNING
p= {}
p["DT_MS"] = 1.0
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["NUM_SAMPLES"] = 1

params = {}
params["NUM_INPUT"] = 80
params["NUM_HIDDEN"] = 512
params["NUM_OUTPUT"] = 35

#use_slurm_host() # without this line, system always chose "kp" partition
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

# Z is speaker ID
X_train, Y_train, Z_train, X_test, Y_test, Z_test = utils.load_data_SHD(p,num_samples=p["NUM_SAMPLES"])
print(X_train.shape)
print(Y_train.shape)
print(Z_train.shape)

x_train, y_train, x_test, y_test = GSC_utils.load_data_GSC(dataset_dir = "/mnt/data0/ts468/data/rawSC/rawSC_80input/",
                                                           network_scale = 1,
                                                           num_samples = 1)
print(x_train.shape)
print(y_train.shape)

# transform some parmeters
tau_mem_fac = 1.0-np.exp(-p["DT_MS"]/p["TAU_MEM"])
tau_mem_fac_int = int(np.round(tau_mem_fac*(2**12)))
tau_syn_fac = 1.0-np.exp(-p["DT_MS"]/p["TAU_SYN"])
tau_syn_fac_int = int(np.round(tau_syn_fac*(2**12)))

print(os.listdir())
# load connections
w_i2h = np.load("lava-implementation/quant8-Conn_Pop0_Pop1-g.npy")
w_i2h = w_i2h.reshape((params.get("NUM_INPUT"), params.get("NUM_HIDDEN"))).T
w_i2h *= tau_mem_fac
if do_plots:
    plt.figure()
    plt.hist(w_i2h)
w_h2h = np.load("lava-implementation/quant8-Conn_Pop1_Pop1-g.npy")
w_h2h = w_h2h.reshape((params.get("NUM_HIDDEN"), params.get("NUM_HIDDEN"))).T
w_h2h *= tau_mem_fac
w = np.hstack([w_i2h,w_h2h])
w_2h_fac, mn, mx = utils.rescale_factor(w,weight_bits)
w_i2h[w_i2h > mx] = mx
w_i2h[w_i2h < mn] = mn
w_i2h_int = np.round(w_i2h*w_2h_fac).astype(np.int8)
print(f"i2h: mn == {np.amin(w_i2h_int)}, mx == {np.amax(w_i2h_int)}")
if do_plots:
    plt.figure()
    plt.hist(w_i2h_int)
    plt.savefig("output_1.png")
w_h2h[w_h2h > mx] = mx
w_h2h[w_h2h < mn] = mn
w_h2h_int = np.round(w_h2h*w_2h_fac).astype(np.int8)
print(f"h2h: mn == {np.amin(w_h2h_int)}, mx == {np.amax(w_h2h_int)}")

w_h2o = np.load("lava-implementation/quant8-Conn_Pop1_Pop2-g.npy")
w_h2o = w_h2o.reshape((params.get("NUM_HIDDEN"), params.get("NUM_OUTPUT"))).T
w_h2o *= tau_mem_fac
w_2o_fac, mn, mx = utils.rescale_factor(w_h2o,weight_bits)
w_2o_fac /= 2.0
mn *= 2.0
mx *= 2.0
w_h2o[w_h2o > mx] = mx
w_h2o[w_h2o < mn] = mn
w_h2o_int = np.round(w_h2o*w_2o_fac).astype(np.int8)
print(f"h2o: mn == {np.amin(w_h2o_int)}, mx == {np.amax(w_h2o_int)}")

vth_hid = w_2h_fac
vth_hid_int = int(np.round(vth_hid))

X_test= np.hstack(X_test)
print(X_test.shape)
# Create processes
input = RingBuffer(data=X_test)

py2nx_inp = PyToNxAdapter(shape=(X_test.shape[0],))

hidden = LIFReset(shape=(1024, ),                         # Number and topological layout of units in the process
                  vth=vth_hid_int,                             # Membrane threshold
                  dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                  du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                  bias_mant=0,           # Bias added to the membrane voltage in every timestep
                  name="hidden",
                  reset_interval=1024,
                  log_config=log_config)

output = LIFReset(shape=(20, ),                         # Number and topological layout of units in the process
                  vth=2**30,                             # Membrane threshold
                  dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                  du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                  bias_mant=0,           # Bias added to the membrane voltage in every timestep
                  name="output",
                  reset_interval=1024,
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

out = SpikeOut(shape=(20,), buffer=X_test.shape[1])
v_read_adapter = StateReader(shape=(20,), offset=0, interval=1)
v_read_adapter.connect_var(output.v)
v_read_adapter.out.connect(out.a_in)

# monitor outputs

# monitor_output = Monitor()
num_steps = int(1000/p["DT_MS"])

# monitor_output.probe(output.v, X_test.shape[1])

# run something
run_condition = RunSteps(num_steps=num_steps)
#run_cfg = Loihi1SimCfg(select_tag="floating_pt")
# run_cfg = Loihi1SimCfg(select_tag="fixed_pt")

loihi2hw_exception_map = {
            SpikeOut: PyReceiveModelFixed,
        }

run_cfg = Loihi2HwCfg(exception_proc_model_map=loihi2hw_exception_map,callback_fxs=[probe_v])
# run_config.select(conn_proto_readout,[PyDenseModelBitAcc])
output._log_config.level = logging.INFO

n_sample = X_test.shape[1]//num_steps
for i in tqdm(range(n_sample)):
    output.run(condition=run_condition, run_cfg=run_cfg)

#print("Weight: ", probe_v.time_series[::10])
#output_v = out.data.get()
print(probe_v.time_series[:num_steps])
output.stop()
output_v = probe_v.time_series.reshape(num_steps,20).T
print(output_v.shape)
print(output_v)
# output_v = monitor_output.get_data()
good = 0
for i in range(n_sample):
    out_v = output_v[:,i*num_steps:(i+1)*num_steps]
    print(np.sum(output_v))
    sum_v = np.sum(out_v,axis=1)
    print(sum_v)
    print(Y_test[i])
    pred = np.argmax(sum_v)
    # print(f"Pred: {pred}, True:{Y_test[i]}")
    if pred == Y_test[i]:
        good += 1
    if do_plots:
        plt.figure()
        plt.plot(out_v.T)
        plt.show()

print(f"test accuracy: {good/n_sample*100}")


# Got 87.1% from the pre-trained model that had 89% in GeNN
