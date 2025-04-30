import numpy as np
from lava.proc.lif.process import LIF
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

def rescale_factor(w, bits):
    rng = float(2**(bits-1))
    mx = max(np.percentile(w,99), np.percentile(-w,99))
    fac = (rng-1)/mx
    mn = -(rng-2)/fac
    return (fac, mn, mx)

def Lava_Run(scale_val, vthres):

    params = {}
    params["DT_MS"] = 1.0
    params["TAU_MEM"] = 20.0
    params["TAU_SYN"] = 2.0
    params["num_samples"] = 1 #int(11005 / 8)
    params["sample_id"] = 0     #sample used for graph generation (starting at 0, < num_samples)

    params["timesteps"] = 2000

    params["NUM_INPUT"] = 80
    params["NUM_HIDDEN"] = 512
    params["NUM_OUTPUT"] = 35

    # toggle to record spikes, useful for debugging, but memory intensive
    params["record_network_ih_activity"] =  True

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
    w_2o_fac /= scale_val
    mn *= scale_val
    mx *= scale_val
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

    hidden = LIF(shape=(params["NUM_HIDDEN"], ),                         # Number and topological layout of units in the process
                    vth=vth_hid_int,                             # Membrane threshold
                    dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                    du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                    bias_mant=0,           # Bias added to the membrane voltage in every timestep
                    name="hidden")

    output = LIF(shape=(params["NUM_OUTPUT"], ),                         # Number and topological layout of units in the process
                    vth=2**vthres,                             # Membrane threshold
                    dv=tau_mem_fac_int,                              # Inverse membrane time-constant
                    du=tau_syn_fac_int,                              # Inverse synaptic time-constant
                    bias_mant=0,           # Bias added to the membrane voltage in every timestep
                    name="output")

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

    monitor_output = Monitor()
    monitor_output.probe(output.v, the_x.shape[1])

    # monitor outputs
    num_steps = int(params["timesteps"]/params["DT_MS"])
    assert(num_steps == 2000)

    # run something
    run_condition = RunSteps(num_steps=num_steps)
    run_cfg = Loihi2SimCfg(select_tag="fixed_pt") # changed 1 -> 2

    for i in tqdm(range(params["num_samples"])):
        output.run(condition=run_condition, run_cfg=run_cfg)

    output_v = monitor_output.get_data()
    good = 0

    for i in range(params["num_samples"]):
        out_v = output_v["output"]["v"][i*num_steps:(i+1)*num_steps,:]
        sum_v = np.sum(out_v,axis=0)
        pred = np.argmax(sum_v)
        # print(f"Pred: {pred}, True:{Y_test[i]}")
        if pred == the_y[i]:
            good += 1
        if do_plots:
            plt.figure(figsize=(8, 5), dpi=80)
            plt.plot(out_v)
            plt.title(f"Output layer voltage with vth of 2^{vthres} and scale of {scale_val}")
            plt.ylabel("voltage (v)")
            plt.xlabel("timesteps")
            #plt.axhline(2**vthres, color = "red", linestyle = "dashed")

            plt.savefig(f"image_output_noReset_sim/scan_vth_{vthres}_scale_{scale_val}.png")

    output.stop()


Lava_Run(scale_val = 2, vthres = 15)