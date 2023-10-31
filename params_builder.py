import json
import os
import sys
#combinations
combinations = {}
directory_name = "jade_test_params_02"
params = {}

previous_d = os.getcwd()
os.chdir(os.path.expanduser("~/PhD/Intel-Neuromorphic-Research-Project/"))
print(os.getcwd())

try:
    os.mkdir(directory_name)
    print("directory made")
except:
    pass

os.chdir(previous_d) #fix this TODO:

if len(sys.argv) != 2:
    print("running local parameters")
    print("error: please pass json file")
    exit(1)

    params["NUM_INPUT"] = 40
    params["NUM_HIDDEN"] = 512 #256
    params["NUM_OUTPUT"] = 20
    params["BATCH_SIZE"] = 256
    params["INPUT_FRAME_TIMESTEP"] = 20#20
    params["INPUT_SCALE"] = 0.00099 #0.008
    params["NUM_EPOCH"] = 10
    params["NUM_FRAMES"] = 80
    params["verbose"] = True
    params["debug"] = False
    params["lr"] = 0.008 #0.009 @ 96.97 #0.008 @ 97.12 
    params["dt"] = 1

    params["reg_lambda_lower"] = 1e-12
    params["reg_lambda_upper"] = 1e-12
    params["reg_nu_upper"] = 20

    #weights
    params["hidden_w_mean"] = 0.0
    params["hidden_w_sd"] = 3.5
    params["output_w_mean"] = 3.0
    params["output_w_sd"] = 1.5 

    # Augmentation
    params["aug_combine_images"] = True
    params["aug_swap_pixels"] = True
    params["aug_swap_pixels_kSwap"] = 1
    params["aug_swap_pixels_pSwap"] = 0.2
    params["aug_swap_pixels_tSwap"] = 0.1

    params["cross_validation"] = True
    params["cross_validation_run_all"] = True

else:
    print("running passed arguments")
    with open(sys.argv[1], "r") as f:
        params = json.load(f)

combinations["aug_swap_pixels_kSwap"] = [1, 2, 3, 4]

# Writing to sample.json
# horrid solution TODO: improve this code to support cross combinations
for c_count, c in enumerate(combinations):
    for i_count, i in enumerate(combinations.get(c)):
        params[c] = i
        params["output_dir"] = params.get("output_dir")[:13] + str(i_count + (c_count * len(combinations)))

        json_object = json.dumps(params, indent = 4)
        print(json_object)
        with open(f"{directory_name}/params_{i_count + (c_count * len(combinations))}.json", "w") as outfile:
            outfile.write(json_object)

