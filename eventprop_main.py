#export CUDA_PATH=/usr/local/cuda
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from tqdm import trange
import os
import pickle
import copy
import math
import json
from numba import cuda

from ml_genn import InputLayer, Layer, SequentialNetwork, Network, Population, Connection
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput, LeakyIntegrateFireInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from time import perf_counter

import nvsmi

from ml_genn.utils.data import (calc_latest_spike_time, linear_latency_encode_data)
from ml_genn.compilers.event_prop_compiler import default_params

import random
import librosa

import augmentation_tools

def eventprop(params):
    """
    Function to run hd classification using eventprop
    Parameters:
      params - a dictionary containing all parameters
    """
    
    # change dir for readout files
    # lazy fix until a solution can be implemented with ml_genn to support output file directory change
    try:
        os.makedirs(params.get("output_dir") + params.get("sweeping_suffix"))
    except:
        pass

    os.chdir(params.get("output_dir") + params.get("sweeping_suffix"))

    # Load dataset
    x_train = np.load(os.path.expanduser(params.get("dataset_directory")) + "training_x_data.npy")
    y_train = np.load(os.path.expanduser(params.get("dataset_directory")) + "training_y_data.npy")
    
    x_test = np.load(os.path.expanduser(params.get("dataset_directory")) + "testing_x_data.npy")
    y_test = np.load(os.path.expanduser(params.get("dataset_directory")) + "testing_y_data.npy")
    
    #TODO: Redundant? any point to have a NUM_INPUT if data can be obtained through dataset shape
    assert x_train.shape[1] == params.get("NUM_INPUT"), "dataset input size doesn't match passed input parameter size"
    
    if params.get("NETWORK_SCALE") < 1:
        assert len(x_train) == len(y_train)
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        print(f"original network size: {len(x_train)}")
        x_train = x_train[:int(len(x_train) * params.get("NETWORK_SCALE"))]
        y_train = y_train[:int(len(y_train) * params.get("NETWORK_SCALE"))]
        print(f"reduced network size: {len(x_train)}")
        print("!! network reduced")

    if params.get("cross_validation"):
        training_details = pd.read_csv(os.path.expanduser(params.get("dataset_directory")) + "training_details.csv")
        testing_details = pd.read_csv(os.path.expanduser(params.get("dataset_directory")) + "testing_details.csv")

    training_images = np.swapaxes(x_train, 1, 2) 
    testing_images = np.swapaxes(x_test, 1, 2) 
 
    training_images = training_images + abs(np.floor(training_images.min()))
    testing_images = testing_images + abs(np.floor(testing_images.min()))

    training_labels = y_train
    testing_labels = y_test
    
    # adding validation data if exists
    validation_images = np.array([])
    validation_labels = np.array([])
    if os.path.isfile(os.path.expanduser(params.get("dataset_directory")) + "validation_y_data.npy"):
        print("!! validation dataset loaded successfully")
        x_validation = np.load(os.path.expanduser(params.get("dataset_directory")) + "validation_x_data.npy")
        y_validation = np.load(os.path.expanduser(params.get("dataset_directory")) + "validation_y_data.npy")
        
        validation_images = np.swapaxes(x_validation, 1, 2) 
        validation_images = validation_images + abs(np.floor(validation_images.min()))
        
        validation_labels = y_validation
    
    # readout class
    class CSVTrainLog(Callback):
        def __init__(self, filename, output_pop, resume):
            # Create CSV writer
            self.file = open(filename, "a" if resume else "w")
            self.csv_writer = csv.writer(self.file, delimiter=",")

            # Write header row if we're not resuming from an existing training run
            if not resume:
                self.csv_writer.writerow(["Epoch", "Num trials", "Number correct", "accuracy", "Time", "memory"])

            self.output_pop = output_pop

        def on_epoch_begin(self, epoch):
            self.start_time = perf_counter()

        def on_epoch_end(self, epoch, metrics):
            processes = nvsmi.get_gpu_processes()
            process = next(p for p in processes if p.pid == os.getpid())
            m = metrics[self.output_pop]
            self.csv_writer.writerow([epoch, 
                                    m.total, 
                                    m.correct,
                                    m.correct / m.total,
                                    perf_counter() - self.start_time,
                                    process.used_memory])
            self.file.flush()
            
    # Create sequential model
    serialiser = Numpy(params.get("model_description"))
    network = Network(default_params)
    
    with network:
        # Populations
        input = Population(LeakyIntegrateFireInput(v_thresh=1, 
                                                tau_mem=20,    
                                                input_frames=params.get("NUM_FRAMES"), 
                                                input_frame_timesteps=params.get("INPUT_FRAME_TIMESTEP")),
                            params.get("NUM_INPUT"), 
                            record_spikes = True)
        
        hidden = Population(LeakyIntegrateFire(v_thresh=1.0, 
                                        tau_mem=20.0),
                    params.get("NUM_HIDDEN"), 
                    record_spikes=True)
        
        output = Population(LeakyIntegrate(tau_mem=20.0, 
                                    readout="avg_var_exp_weight"),
                    params.get("NUM_OUTPUT"), 
                    record_spikes=True)

        # Connections
        Connection(input, hidden, Dense(Normal(mean = params.get("input_hidden_w_mean"), 
                                               sd = params.get("input_hidden_w_sd"))),
                    Exponential(2.0))
        
        if params.get("recurrent"):
            Connection(hidden, hidden, Dense(Normal(mean = params.get("hidden_hidden_w_mean"), 
                                                    sd = params.get("hidden_hidden_w_sd"))),
                    Exponential(2.0))
        
        Connection(hidden, output, Dense(Normal(mean = params.get("hidden_output_w_mean"),
                                    sd = params.get("hidden_output_w_sd"))),
                   Exponential(2.0))
        
    compiler = EventPropCompiler(example_timesteps = params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"),
                            losses="sparse_categorical_crossentropy",
                            optimiser=Adam(params.get("lr")), 
                            batch_size = params.get("BATCH_SIZE"),
                            reg_lambda_lower = params.get("reg_lambda_lower"),
                            reg_lambda_upper = params.get("reg_lambda_upper"),
                            reg_nu_upper = params.get("reg_nu_upper"),
                            dt = params.get("dt"),
                            max_spikes=1500)

    compiled_net = compiler.compile(network)
    
    if params.get("cross_validation"):
        
        if params.get("verbose"): print(training_details.head())
        speaker_id = np.sort(training_details.Speaker.unique())
        speaker_id = speaker_id = speaker_id.astype('int8')  #np where is fussy with int
    
        if params.get("verbose"): print(np.sort(testing_details.Speaker.unique()))
        
        # Create sequential model
        serialisers = []
        for s in speaker_id:
            serialisers.append(Numpy(f"serialiser_{s}"))
        
        print(speaker_id)
        print(len(serialisers))
        
        speaker = list(training_details.loc[:, "Speaker"])
        
        for count, speaker_left in enumerate(speaker_id):
            #reset gpu memory
            device = cuda.get_current_device()
            device.reset()
            
            train= np.where(speaker != speaker_left)[0]
            evalu= np.where(speaker == speaker_left)[0]
            train_spikes= np.array([ training_images[i] for i in train ])
            eval_spikes= np.array([ training_images[i] for i in evalu ])
            train_labels= [ training_labels[i] for i in train ]
            eval_labels= [ training_labels[i] for i in evalu ]
            
            print(f"speaker {speaker_left} of {len(speaker_id)}")
            print("\ncount", count)

            # Augmentation
            if params.get("aug_combine_images"):
                train_spikes, train_labels = augmentation_tools.combine_two_images_and_concatinate(copy.deepcopy(train_spikes), 
                                                                                                   train_labels)

            with compiled_net:
                # Evaluate model on numpy dataset
                callbacks = [Checkpoint(serialiser), 
                             CSVTrainLog(f"train_output_{speaker_left}.csv", 
                                         output,
                                         False)]
                
                if params.get("verbose"):
                    callbacks.append("batch_progress_bar")
                
                if params.get("debug"):
                    print("!!!    debug")
                    callbacks.append(SpikeRecorder(hidden, 
                                            key = "hidden_spike_counts", 
                                            record_counts = True,
                                            example_filter = list(range(7, # random sample from trial, in this case the trial chosen is 7000
                                                                        params.get("NUM_EPOCH") * int(math.ceil((len(x_train) * 0.9) / params.get("BATCH_SIZE"))) * params.get("BATCH_SIZE"), 
                                                                        int(math.ceil((len(x_train) * 0.9) / params.get("BATCH_SIZE"))) * params.get("BATCH_SIZE")))))

                if params.get("record_all_hidden_spikes"):
                    callbacks.append(SpikeRecorder(hidden, 
                                            key = "hidden_spike_counts_unfiltered", 
                                            record_counts = True))
                    
                for e in trange(params.get("NUM_EPOCH")):
                    e_train_spikes = copy.deepcopy(train_spikes)
                    #complete augmentation\
                    if params.get("aug_swap_pixels"):
                        e_train_spikes = augmentation_tools.pixel_swap(copy.deepcopy(train_spikes),
                                                                    params.get("aug_swap_pixels_kSwap"),
                                                                    params.get("aug_swap_pixels_pSwap"),
                                                                    params.get("aug_swap_pixels_tSwap"))

                    metrics, metrics_val, cb_data_training, cb_data_validation = compiled_net.train({input: e_train_spikes * params.get("INPUT_SCALE")},
                                                                                                {output: train_labels},
                                                                                                start_epoch = e, 
                                                                                                num_epochs = 1,
                                                                                                shuffle = not(params.get("debug")),
                                                                                                callbacks=callbacks,
                                                                                                validation_x= {input: eval_spikes * params.get("INPUT_SCALE")},
                                                                                                validation_y= {output: eval_labels})
        
        if params.get("cross_validation_run_all"): 
            print("\n\nrun across all values ")
            combined_serialiser = Numpy("serialiser_all")

            with compiled_net:
                # Evaluate model on numpy dataset
                
                callbacks = [Checkpoint(combined_serialiser), 
                             CSVTrainLog(f"train_output_combined.csv", 
                                         output,
                                         False)]
                
                if params.get("verbose"):
                    callbacks.append("batch_progress_bar")
                
                if params.get("debug"):
                    print("!!!    debug")
                    callbacks.append(SpikeRecorder(hidden, 
                                            key = "hidden_spike_counts", 
                                            record_counts = True,
                                            example_filter = list(range(7000, # random sample from trial, in this case the trial chosen is 7000
                                                                        params.get("NUM_EPOCH") * int(math.ceil((len(x_train) * 0.9) / params.get("BATCH_SIZE"))) * params.get("BATCH_SIZE"), 
                                                                        int(math.ceil((len(x_train) * 0.9) / params.get("BATCH_SIZE"))) * params.get("BATCH_SIZE")))))

                if params.get("record_all_hidden_spikes"):
                    callbacks.append(SpikeRecorder(hidden, 
                                            key = "hidden_spike_counts_unfiltered", 
                                            record_counts = True))
                    
                metrics, metrics_val, cb_data_training, cb_data_validation = compiled_net.train({input: training_images * params.get("INPUT_SCALE")},
                                                                {output: training_labels},
                                                                num_epochs = params.get("NUM_EPOCH"), 
                                                                shuffle = not(params.get("debug")),
                                                                callbacks=callbacks,
                                                                validation_x= {input: eval_spikes * params.get("INPUT_SCALE")},
                                                                validation_y= {output: eval_labels})
        
        # evaluate
        network.load((params.get("NUM_EPOCH") - 1,), serialiser)

        compiler = InferenceCompiler(evaluate_timesteps = params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"),
                                    reset_in_syn_between_batches=True,
                                    batch_size = params.get("BATCH_SIZE"))
        compiled_net = compiler.compile(network)

        with compiled_net:
            
            if params.get("verbose"):
                callbacks = ["batch_progress_bar", 
                            Checkpoint(serialiser), 
                            SpikeRecorder(input, key="input_spikes"), 
                            SpikeRecorder(hidden, key="hidden_spikes"),
                            #SpikeRecorder(output, key="output_spikes"),
                            VarRecorder(output, "v", key="v_output")]
            else:
                callbacks = [Checkpoint(serialiser)]
        
            metrics, cb_data = compiled_net.evaluate({input: testing_images * params.get("INPUT_SCALE")},
                                                    {output: testing_labels},
                                                    callbacks = callbacks)
        
    #if params.get("cross_validation") == False:
    else:
        with compiled_net:
            # Evaluate model on numpy dataset
            start_time = perf_counter()
            
            callbacks = [CSVTrainLog(f"train_output.csv", 
                                    output,
                                    False),
                        Checkpoint(serialiser)]
            
            if params.get("verbose"):
                callbacks.append("batch_progress_bar")
                
            if params.get("debug"):
                print("!!!    debug")
                callbacks.append(SpikeRecorder(hidden, 
                                        key = "hidden_spike_counts", 
                                        record_counts = True,
                                        example_filter = list(range(70, # random sample from trial, in this case the trial chosen is 7000 # x1.0 is for validation split!
                                                                    params.get("NUM_EPOCH") * int(math.ceil((len(x_train) * 1) / params.get("BATCH_SIZE"))) * params.get("BATCH_SIZE"), 
                                                                    int(math.ceil((len(x_train) * 1) / params.get("BATCH_SIZE"))) * params.get("BATCH_SIZE")))))

            if params.get("record_all_hidden_spikes"):
                callbacks.append(SpikeRecorder(hidden, 
                                        key = "hidden_spike_counts_unfiltered", 
                                        record_counts = True))

                
            if not bool(validation_images.any()):    
                metrics, metrics_val, cb_data_training, cb_data_validation = compiled_net.train({input: testing_images * params.get("INPUT_SCALE")},
                                                                                                {output: testing_labels},
                                                                                                num_epochs = params.get("NUM_EPOCH"),
                                                                                                shuffle = not(params.get("debug")),
                                                                                                validation_split = 0.1,
                                                                                                callbacks = callbacks)    
                
            else:
                metrics, metrics_val, cb_data_training, cb_data_validation = compiled_net.train({input: testing_images * params.get("INPUT_SCALE")},
                                                                                                {output: testing_labels},
                                                                                                num_epochs = params.get("NUM_EPOCH"),
                                                                                                shuffle = not(params.get("debug")),
                                                                                                callbacks = callbacks,
                                                                                                validation_x = {input: validation_images * params.get("INPUT_SCALE")},
                                                                                                validation_y = {output: validation_labels})  

            end_time = perf_counter()
            print(f"Time = {end_time - start_time}s")
            
        if params.get("debug"):
            # pickle serialisers
            with open('serialisers.pkl', 'wb') as f:
                pickle.dump(serialiser, f)

            # save hidden training spike counts
            with open(f'hidden_training_spike_counts.npy', 'wb') as f:     
                hidden_spike_counts = np.array(cb_data_training["hidden_spike_counts"], dtype=np.int16)
                np.save(f, hidden_spike_counts)
                
            # save hidden training spike counts
            with open(f'hidden_validation_spike_counts.npy', 'wb') as f:     
                hidden_spike_counts = np.array(cb_data_validation["hidden_spike_counts"], dtype=np.int16)
                np.save(f, hidden_spike_counts)
            
            # get hidden spikes if param is true
            if params.get("record_all_hidden_spikes"):
                # save all hidden spike counts
                with open(f'hidden_spike_counts_unfiltered.npy', 'wb') as f:     
                    hidden_spike_counts = np.array(cb_data_training["hidden_spike_counts_unfiltered"], dtype=np.int16)
                    np.save(f, hidden_spike_counts)

            # save parameters for reference
            json_object = json.dumps(params, indent = 4)
            with open("params.json", "w") as outfile:
                outfile.write(json_object)
                
            
        # evaluate
        if params.get("evaluate"):
            network.load((params.get("NUM_EPOCH") - 1,), serialiser)

            compiler = InferenceCompiler(evaluate_timesteps = params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"),
                                        reset_in_syn_between_batches=True,
                                        batch_size = params.get("BATCH_SIZE"))
            compiled_net = compiler.compile(network)

            with compiled_net:
                if params.get("verbose"):
                    callbacks = ["batch_progress_bar", 
                                Checkpoint(serialiser),
                                SpikeRecorder(input, key="input_spikes"), 
                                SpikeRecorder(hidden, key="hidden_spikes"),
                                VarRecorder(output, "v", key="v_output")]
                else:
                    callbacks = [Checkpoint(serialiser)]
            
                metrics, cb_data = compiled_net.evaluate({input: training_images * params.get("INPUT_SCALE")},
                                                        {output: training_labels},
                                                        callbacks = callbacks)
            
    if params.get("verbose") and params.get("evaluate"):
        # cannot print verbose whilst requesting just accuracy
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('rawHD with EventProp on ml_genn')

        value = random.randint(0, 1000) #len(x_test))

        ax1.scatter(cb_data["hidden_spikes"][0][value], 
                    cb_data["hidden_spikes"][1][value], s=1)
        ax1.set_xlabel("Time [ms]")
        ax1.set_ylabel("Neuron ID")
        ax1.set_title("Hidden")
        ax1.set_xlim(0, params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"))
        ax1.set_ylim(0, params.get("NUM_HIDDEN"))

        ax2.scatter(cb_data["input_spikes"][0][value], 
                    cb_data["input_spikes"][1][value], s=1)
        ax2.set_xlabel("Time [ms]")
        ax2.set_ylabel("Neuron ID")
        ax2.set_title("Input")
        ax2.set_xlim(0, params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"))
        ax2.set_ylim(0, params.get("NUM_INPUT"))

        ax3.plot(cb_data["v_output"][value])
        ax3.set_xlabel("Time [ms]")
        ax3.set_ylabel("voltage (v)")
        ax3.set_title("Output voltage")
        ax3.set_xlim(0, params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"))
        #ax3.set_ylim(0, params.get("NUM_INPUT"))

        sr = 22050
        img = librosa.display.specshow(x_train[value], 
                                x_axis='time', 
                                y_axis='mel', 
                                sr=sr, 
                                cmap='viridis')
        #fig.colorbar(img, ax = ax4)
        ax4.set_title("mel encoding")

        fig.tight_layout()
        #figure(figsize=(8, 6), dpi=200)
        plt.savefig('activity_across_layers.png')
        plt.clf() 
        
        if params.get("cross_validation"):
            # show accuracy log
            for speaker_left in speaker_id:
        
                data = pd.read_csv(f"train_output_{speaker_left}.csv")
                df = pd.DataFrame(data, columns=['accuracy'])

                accuracy = np.array(df)

                accuracy = accuracy * 100

                validation = []
                training = []

                for i in range(len(accuracy)):
                    if i % 2 == 0:
                        training.append(float(accuracy[i]))
                    else:
                        validation.append(float(accuracy[i]))
                        
                plt.plot(training, label = f"training_{speaker_left}")
            plt.ylabel("accuracy (%)")
            plt.xlabel("epochs")
            plt.ylim(0, 100)
            plt.title("accuracy during training")
            plt.legend()
            #figure(figsize=(8, 6), dpi=200)
            plt.savefig('accuracy_over_time.png')
            plt.clf() 
            
            # show validation log
            for speaker_left in speaker_id:
        
                data = pd.read_csv(f"train_output_{speaker_left}.csv")
                df = pd.DataFrame(data, columns=['accuracy'])

                accuracy = np.array(df)

                accuracy = accuracy * 100

                validation = []
                training = []

                for i in range(len(accuracy)):
                    if i % 2 == 0:
                        training.append(float(accuracy[i]))
                    else:
                        validation.append(float(accuracy[i]))
                        
                plt.plot(validation, label = f"validation_{speaker_left}")
            plt.ylabel("accuracy (%)")
            plt.xlabel("epochs")
            plt.ylim(0, 100)
            plt.title("validation during training")
            plt.legend()
            #figure(figsize=(8, 6), dpi=200)
            plt.savefig('validation_over_time.png')
            plt.clf() 
            
        else:
            data = pd.read_csv("train_output.csv")
            df = pd.DataFrame(data, columns=['accuracy'])

            accuracy = np.array(df)

            accuracy = accuracy * 100

            validation = []
            training = []

            for i in range(len(accuracy)):
                if i % 2 == 0:
                    training.append(float(accuracy[i]))
                else:
                    validation.append(float(accuracy[i]))
                                
            plt.plot(training, label = "training")
            plt.plot(validation, label = "validation")
            plt.ylabel("accuracy (%)")
            plt.ylim(0, 100)
            plt.xlabel("epochs")
            plt.title("accuracy during training")
            #figure(figsize=(8, 6), dpi=200)
            plt.legend()
            plt.savefig('v&a_over_time.png')
            plt.clf() 

    # reset directory
    os.chdir("..")

    # if sweeping, save params and accuracy to csv
    does_summary_file_exist = os.path.exists("summary.csv")
    # Create CSV writer
    file = open("summary.csv", "a" if does_summary_file_exist else "w")
    csv_writer = csv.writer(file, delimiter=",")

    # Write header row if we're not resuming from an existing training run
    if not does_summary_file_exist:
        csv_writer.writerow((list(params) + ["accuracy"]))

    csv_writer.writerow((list(params.values()) + [metrics[output].correct / metrics[output].total]))

    return metrics[output].correct / metrics[output].total