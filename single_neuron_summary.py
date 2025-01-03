from matplotlib import pyplot as plt
import numpy as np

from ml_genn import Network, Population, Connection
from ml_genn.callbacks import SpikeRecorder, VarRecorder
from ml_genn.compilers import InferenceCompiler
from ml_genn.connectivity import Dense as genn_Dense
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
from ml_genn.synapses import Exponential

from ml_genn.utils.data import calc_latest_spike_time
from ml_genn.compilers.event_prop_compiler import default_params

from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)

from lava.proc.lif.process import LIF
from lava.proc.dense.process import Dense
from lava.proc.io.source import RingBuffer

from lava.proc.monitor.process import Monitor
from lava.magma.core.run_configs import Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps
from lava.magma.core.run_conditions import RunContinuous

params = {}
params["dt"] = 1.0
params["tau_mem"] = 20.0
params["tau_syn"] = 5.0
params["vth"] = 35
params["timesteps"] = 150
params["weight"] = 10
params["spike_times"] = [10]
params["genn_spike_injection_current"] = 0.04420300925285642

#tau_mem_fac = 1.0-np.exp(-dt/tau_mem)
#tau_syn_fac = 1.0-np.exp(-dt/tau_syn)


def numpy_LIF_output(params, spike_injection_value = 1):

    alpha = np.exp(-params["dt"]/params["tau_syn"])
    beta = np.exp(-params["dt"]/params["tau_mem"])

    I = np.zeros(params["timesteps"] + 1)
    U = np.zeros(params["timesteps"] + 1)
    S = np.zeros(params["timesteps"] + 1)
    X = np.zeros(params["timesteps"] + 1)
    
    for i in params["spike_times"]:
        X[i] = spike_injection_value #0.04420300925285642

    for t in range(params["timesteps"]):
        I[t + 1] = (alpha * I[t]) + (params["weight"] * X[t])
        U[t + 1] = (beta * U[t]) + I[t + 1]
        # reset code
        if U[t + 1] > params["vth"]: 
            S[t + 1] = 1
            U[t + 1] = 0 #-= vth
            
    return U, S

def genn_floating_output(params, spike_injection_value = 1):
    x = []
    for i in params["spike_times"]:
        x.append((0, i, 1))
    input_spikes = np.array(x, 
                            dtype = ([('x', np.int8), 
                                    ('t', np.uint16),
                                    ('p', np.int8)]))
    
    # Preprocess
    x_test_spikes = []
    y_test_spikes = [0]

    x_test_spikes.append(preprocess_tonic_spikes(input_spikes, 
                                                    input_spikes[0].dtype.names,
                                                    (1, 1, 1),
                                                    time_scale = 1))

    # Determine max spikes and latest spike time
    max_spikes = calc_max_spikes(x_test_spikes)
    latest_spike_time = calc_latest_spike_time(x_test_spikes)
    print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")
    
    network = Network(default_params)

    with network:
        # Populations
        input = Population(SpikeInput(max_spikes = 10),
                        1,
                        record_spikes=True)
        
        hidden = Population(LeakyIntegrateFire(v_thresh=params["vth"], 
                                        tau_mem=params["tau_mem"]),
                    1, 
                    record_spikes=True)
        
        output = Population(LeakyIntegrate(tau_mem=params["tau_mem"], 
                                    readout="avg_var_exp_weight"),
                    1, 
                    record_spikes=True)
        
        Connection(input, hidden, genn_Dense(weight = np.array([[params["weight"]]])),
                    Exponential(5))
        
        Connection(hidden, output, genn_Dense(weight = np.array([[params["weight"]]])),
                    Exponential(5))
        
        
    compiler = InferenceCompiler(evaluate_timesteps = params["timesteps"],
                            reset_in_syn_between_batches=True,
                            batch_size = 1)

    compiled_net = compiler.compile(network)
    
    with compiled_net:

        callbacks = ["batch_progress_bar",
                    SpikeRecorder(input, 
                                key = "input_spikes",
                                example_filter = 0),
                    SpikeRecorder(hidden,
                                key = "hidden_spikes",
                                example_filter = 0),
                    VarRecorder(hidden, 
                                var = "v",
                                key = "hidden_voltages",
                                example_filter = 0)]
        
        metrics, cb_data = compiled_net.evaluate({input: x_test_spikes}, {output: y_test_spikes}, callbacks = callbacks)
        


    return cb_data["hidden_voltages"][0], cb_data["hidden_spikes"][0][0]

def lava_floating_output(params, spike_injection_value = 1):
    
    input_spikes = np.zeros(shape = (1, params["timesteps"]))
    for i in params["spike_times"]:
        input_spikes[0][i] = 0.04420300925285642
    
    input = RingBuffer(data = input_spikes)

    # Create processes
    lif1 = LIF(shape=(1, ),                         
            vth=params["vth"],
            dv=1.0-np.exp(-params["dt"]/params["tau_mem"]),
            du=1.0-np.exp(-params["dt"]/params["tau_syn"]),
            name="lif1")

    dense = Dense(weights=np.array([[params["weight"]]]),
                name='dense')

    input.s_out.connect(dense.s_in)
    dense.a_out.connect(lif1.a_in)
    
    monitor_lif1 = Monitor()
    monitor_lif1.probe(lif1.v, params["timesteps"])

    monitor_lif1_s = Monitor()
    monitor_lif1_s.probe(lif1.s_out, params["timesteps"])

    # execute
    run_condition = RunContinuous()
    run_condition = RunSteps(num_steps=params["timesteps"])
    run_cfg = Loihi2SimCfg(select_tag="floating_pt")
    lif1.run(condition=run_condition, run_cfg=run_cfg)
    
    data_lif1 = monitor_lif1.get_data()
    data_lif1_s = monitor_lif1_s.get_data()
    
    return data_lif1.get("lif1").get("v"), np.where(data_lif1_s.get("lif1").get("s_out") > 0)[0]

def lava_fixed_output(params, spike_injection_value = 1):
    
    input_spikes = np.zeros(shape = (1, params["timesteps"]))
    for i in params["spike_times"]:
        input_spikes[0][i] = spike_injection_value
        
    tau_mem_fac = 1.0-np.exp(-params["dt"]/params["tau_mem"])
    tau_mem_fac_int = int(np.round(tau_mem_fac*(2**12)))

    tau_syn_fac = 1.0-np.exp(-params["dt"]/params["tau_syn"])
    tau_syn_fac_int = int(np.round(tau_syn_fac*(2**12)))
    
    input = RingBuffer(data = input_spikes)

    # Create processes
    lif1 = LIF(shape=(1, ),                         # Number and topological layout of units in the process
            vth=params["vth"],                             # Membrane threshold
            dv=tau_mem_fac_int,                              # Inverse membrane time-constant
            du=tau_syn_fac_int,                              # Inverse synaptic time-constant
            name="lif1")

    dense = Dense(weights=np.array([[10]], dtype = "int8"),     # Initial value of the weights, chosen randomly
                name='dense')

    input.s_out.connect(dense.s_in)
    dense.a_out.connect(lif1.a_in)

    monitor_lif1 = Monitor()
    monitor_lif1.probe(lif1.v, params["timesteps"])

    monitor_lif1_s = Monitor()
    monitor_lif1_s.probe(lif1.s_out, params["timesteps"])

    # execute
    run_condition = RunContinuous()
    run_condition = RunSteps(num_steps=params["timesteps"])
    run_cfg = Loihi2SimCfg(select_tag="fixed_pt")
    lif1.run(condition=run_condition, run_cfg=run_cfg)
    
    data_lif1 = monitor_lif1.get_data()
    data_lif1_s = monitor_lif1_s.get_data()
    
    return data_lif1.get("lif1").get("v"), np.where(data_lif1_s.get("lif1").get("s_out") > 0)[0]

plt.axhline(params["vth"], c = "r", label = "Threshold", linestyle = "--")

# numpy
numpy_U, numpy_S = numpy_LIF_output(params)
plt.plot(numpy_U, 
         label = "Voltage - numpy",
         linestyle = (0, (1, 1)))
plt.scatter(np.where(numpy_S > 0)[0],
            [[params["vth"]]] * np.where(numpy_S > 0)[0].shape[0],
            c = "C0",
            label = "Spikes - numpy",
            alpha = 0.2)

# genn floating
params["vth"] = 35 * params["genn_spike_injection_current"]
genn_floating_U, genn_floating_S = genn_floating_output(params, 1)
params["vth"] = 35
plt.plot(genn_floating_U / params["genn_spike_injection_current"], 
         label = "Voltage - genn floating")
plt.scatter(genn_floating_S,
            [[params["vth"]]] * genn_floating_S.shape[0],
            c = "C1",
            label = "Spikes - genn floating",
            alpha = 0.2)

# lava floating

lava_floating_U, lava_floating_S = lava_floating_output(params, 0.04420300925285642)
plt.plot(lava_floating_U, 
         label = "Voltage - lava floating",
         linestyle = (1, (1, 1)))
plt.scatter(lava_floating_S,
            [[params["vth"]]] * lava_floating_S.shape[0],
            c = "C2",
            label = "Spikes - lava floating",
            alpha = 0.2)

# lava fixed
lava_fixed_U, lava_fixed_S = lava_fixed_output(params, 1)
fixed_point_rescale_factor = 64             # rescale factor to make fixed pt output similar to floating pt
plt.plot(lava_fixed_U / fixed_point_rescale_factor, label = "Voltage - lava fixed", linestyle = (2, (1, 1)))
plt.scatter(lava_fixed_S,
            [[params["vth"]]] * lava_fixed_S.shape[0],
            c = "C3",
            label = "Spikes - lava fixed",
            alpha = 0.2)

plt.xlim(0, params["timesteps"])
plt.legend()
plt.show()

