import os 
import sys
import json

from HD_eventprop import hd_eventprop

if len(sys.argv) == 2:
    print("running passed arguments")
    with open(sys.argv[1], "r") as f:
        params = json.load(f)

else:
    print("please pass parameters in the form of a json file")
    with open("params.json", "r") as f:
        params = json.load(f)

os.chdir("output")

# nested directory for sweep
if len(params.get("sweeping_suffix")) > 0:
    # change dir for readout files
    try:
        os.makedirs(params.get("output_dir"))
    except:
        pass

    os.chdir(params.get("output_dir"))

accuracy = hd_eventprop(params, 
                        file_path = os.path.expanduser("~/data/rawHD/experimental_2/"))

print(f"accuracy of the network is {accuracy * 100:.2f}%")
