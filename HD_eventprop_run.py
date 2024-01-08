import os 
import sys
import json

from eventprop_main import eventprop

if len(sys.argv) == 2:
    print("running passed arguments")
    with open(sys.argv[1], "r") as f:
        params = json.load(f)

else:
    print("please pass parameters in the form of a json file")
    with open("HD_params.json", "r") as f:
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

accuracy = eventprop(params)

print(f"accuracy of the network is {accuracy * 100:.2f}%")