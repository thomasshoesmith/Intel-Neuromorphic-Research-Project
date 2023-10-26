import os 
import sys
import json

from HD_eventprop import hd_eventprop

if len(sys.argv) == 2:
    print("running passed arguments")
    with open(sys.argv[1], "r") as f:
        params = json.load(f)

else:
    print("running local parameters")
    # constants
    params = {}
    params["NUM_INPUT"] = 40
    params["NUM_HIDDEN"] = 512 #256
    params["NUM_OUTPUT"] = 20
    params["BATCH_SIZE"] = 256
    params["INPUT_FRAME_TIMESTEP"] = 20#20
    params["INPUT_SCALE"] = 0.00099 #0.008
    params["NUM_EPOCH"] = 100
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

accuracy = hd_eventprop(params, 
                        file_path = os.path.expanduser("~/data/rawHD/experimental_2/"),
                        output_dir = "HD_eventprop_cv_aug_both",
                        model_description = "sv_test")

print(f"accuracy of the network is {accuracy * 100:.2f}%")
