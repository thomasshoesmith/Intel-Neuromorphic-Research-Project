import numpy as np
import sys
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
from lava.utils.system import Loihi2
import os
from lava.proc.cyclic_buffer.process import CyclicBuffer
from lava.proc.embedded_io.state import Read as StateReader
#from lava.proc.io.sink import RingBuffer as SpikeOut
from lava.proc.io.sink import PyReceiveModelFixed
from lava.utils.loihi2_state_probes import StateProbe
from lava.proc.embedded_io.spike import PyToNxAdapter


log_config = LogConfig("lava_SHD.log")
log_config.level_console= logging.WARNING
p= {}
p["DT_MS"] = 1.0   # Warning: not freely changeable as we need to have 1024 (or other power of two time steps)
p["TAU_MEM"] = 20.0
p["TAU_SYN"] = 5.0
p["NUM_SAMPLES"] = 1
dur = 1024 # duration of samples in ms - chosen so we get 1024 steps (power of 2)

#use_slurm_host() # without this line, system always chose "kp" partition
os.environ["PATH"] += ":/nfs/ncl/bin:"
os.environ["PARTITION"] = "oheogulch_2h" # _20m _2h (if 2 hours are needed)
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

X_train, Y_train, Z_train, X_test, Y_test, Z_test = utils.load_data_SHD(p,num_samples=p["NUM_SAMPLES"])
print(X_test[0].shape)

# transform some parmeters
tau_mem_fac = 1.0-np.exp(-p["DT_MS"]/p["TAU_MEM"])
tau_mem_fac_int = int(np.round(tau_mem_fac*(2**12)))
print(f"tau_mem_fac_int: {tau_mem_fac_int}")
tau_syn_fac = 1.0-np.exp(-p["DT_MS"]/p["TAU_SYN"])
tau_syn_fac_int = int(np.round(tau_syn_fac*(2**12)))
print(f"tau_syn_fac_int: {tau_syn_fac_int}")

# load connections
w_i2h = np.load("J48b_scan_32_w_input_hidden_best.npy")
w_i2h = w_i2h.reshape((700,1024)).T
w_i2h *= tau_mem_fac
if do_plots:
    plt.figure()
    plt.hist(w_i2h)
w_h2h = np.load("J48b_scan_32_w_hidden0_hidden0_best.npy")
w_h2h = w_h2h.reshape((1024,1024)).T
w_h2h *= tau_mem_fac
w = np.hstack([w_i2h,w_h2h])
w_2h_fac, mn, mx = utils.rescale_factor(w,weight_bits)
w_i2h[w_i2h > mx] = mx
w_i2h[w_i2h < mn] = mn
w_i2h_int = np.round(w_i2h*w_2h_fac).astype(np.int8)
print(f"i2h: mn == {np.amin(w_i2h_int)}, mx == {np.amax(w_i2h_int)}")
#if do_plots:
    #plt.figure()
    #plt.hist(w_i2h_int)
    #plt.show()
w_h2h[w_h2h > mx] = mx
w_h2h[w_h2h < mn] = mn
w_h2h_int = np.round(w_h2h*w_2h_fac).astype(np.int8)
print(f"h2h: mn == {np.amin(w_h2h_int)}, mx == {np.amax(w_h2h_int)}")

w_h2o = np.load("J48b_scan_32_w_hidden_output_best.npy")
w_h2o = w_h2o.reshape((1024,20)).T
w_h2o *= tau_mem_fac
w_2o_fac, mn, mx = utils.rescale_factor(w_h2o,weight_bits)
w_2o_fac /= 8.0
mn *= 8.0
mx *= 8.0
w_h2o[w_h2o > mx] = mx
w_h2o[w_h2o < mn] = mn
w_h2o_int = np.round(w_h2o*w_2o_fac).astype(np.int8)
print(f"h2o: mn == {np.amin(w_h2o_int)}, mx == {np.amax(w_h2o_int)}")

vth_hid = w_2h_fac
vth_hid_int = int(np.round(vth_hid))

X_test= np.hstack(X_test)
ff = X_test[:,0]
rf = X_test[:,1:]
print(ff.shape)
print(rf.shape)
# Create processes
input = CyclicBuffer(first_frame= ff, replay_frames=rf)
dummy_input = RingBuffer(data=X_test)
dummy_adaptor = PyToNxAdapter(shape=(700,))
weight = np.eye(700)
dense = Dense(weights=weight)
        
hidden = LIFReset(shape=(1024, ),                         # Number and topological layout of units in the process
                  vth=vth_hid_int,                             # Membrane threshold
                  dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                  du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                  bias_mant=0,           # Bias added to the membrane voltage in every timestep
                  name="hidden",
                  reset_interval=1024,
                  log_config=log_config)

output = LIFReset(shape=(20, ),                         # Number and topological layout of units in the process
                  vth=2**15,                             # Membrane threshold
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
probe_hid_v = StateProbe(hidden.v)

hid_to_hid = Dense(weights=w_h2h_int,
                   name='hid_to_hid')

hid_to_out = Dense(weights=w_h2o_int,
                   name= 'hid_to_out')

dummy_input.s_out.connect(dummy_adaptor.inp)
dummy_adaptor.out.connect(dense.s_in)
dense.a_out.connect(input.a_in)
input.s_out.connect(in_to_hid.s_in)
in_to_hid.a_out.connect(hidden.a_in)
hidden.s_out.connect(hid_to_hid.s_in)
hidden.s_out.connect(hid_to_out.s_in)
hid_to_hid.a_out.connect(hidden.a_in)
hid_to_out.a_out.connect(output.a_in)

#out = SpikeOut(shape=(20,), buffer=X_test.shape[1])
#v_read_adapter = StateReader(shape=(20,), offset=0, interval=1)
#v_read_adapter.connect_var(output.v)
#v_read_adapter.out.connect(out.a_in)

# monitor outputs

# monitor_output = Monitor()
num_steps = int(dur/p["DT_MS"])
assert(num_steps == 1024)

# monitor_output.probe(output.v, X_test.shape[1])

# run something
run_condition = RunSteps(num_steps=num_steps)
#run_cfg = Loihi1SimCfg(select_tag="floating_pt")
# run_cfg = Loihi1SimCfg(select_tag="fixed_pt")

#loihi2hw_exception_map = {
#            SpikeOut: PyReceiveModelFixed,
#        }

#run_cfg = Loihi2HwCfg(exception_proc_model_map=loihi2hw_exception_map,callback_fxs=[probe_v, probe_hid_v])
run_cfg = Loihi2HwCfg(callback_fxs=[probe_v, probe_hid_v])
#run_config.select(conn_proto_readout,[PyDenseModelBitAcc])
output._log_config.level = logging.INFO

n_sample = X_test.shape[1]//num_steps
print(X_test.shape)
print(n_sample)
for i in tqdm(range(n_sample)):
    output.run(condition=run_condition, run_cfg=run_cfg)

#print("Weight: ", probe_v.time_series[::10])
#output_v = out.data.get()
#print(probe_v.time_series[:num_steps])
output.stop()
print(f"probe_v shape: {probe_v.time_series.shape}")
output_v = probe_v.time_series.reshape(20,num_steps*n_sample).T
hidden_v = probe_hid_v.time_series.reshape(1024,num_steps).T
print(output_v.shape)
print(hidden_v.shape)
#print(output_v)
# output_v = monitor_output.get_data()
good = 0
for i in range(n_sample):
    out_v = output_v[i*num_steps:(i+1)*num_steps,:]
    hid_v = hidden_v[i*num_steps:(i+1)*num_steps,:]
    sum_v = np.sum(out_v,axis=0)
    print(sum_v)
    pred = np.argmax(sum_v)
    print(f"Pred: {pred}, True:{Y_test[i]}")
    if pred == Y_test[i]:
        good += 1
    if do_plots:
        plt.figure()
        plt.plot(out_v)
        plt.figure()
        plt.plot(hid_v)
        plt.show()

print(f"test accuracy: {good/n_sample*100}")


# Got 87.1% from the pre-trained model that had 89% in GeNN
