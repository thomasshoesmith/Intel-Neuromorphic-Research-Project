import sys
import json

if len(sys.argv) != 2:
    print("invalid number of arguments")
    print("running local parameters")
    exit(1)

else:
    print("running passed arguments")


    print(len(sys.argv))

    with open(sys.argv[1], "r") as f:
        params = json.load(f)


    print(params)
    print(type(params))

    print(params.get("NUM_EPOCH"))