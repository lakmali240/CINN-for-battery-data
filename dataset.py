from scipy.io import loadmat
import numpy as np


def load_battery(filename):
    # data is stored as a hierarchy of structs
    # we load with `squeeze_me` to simplify things a little
    data = loadmat(filename, squeeze_me=True)

    # the root variable is the name of the battery set
    # the next key is `cycle`
    # `squeeze_me` ends up with a 0-dim array somehow
    # so we index with and empty tuple to get the data
    events = data[list(data.keys())[-1]]["cycle"][()]

    # only the discharge events have the capacity data
    discharge_capacities = [
        e["data"]["Capacity"][()] for e in events if e["type"] == "discharge"
    ]

    # convert it back to an np.array
    discharge_capacities = np.array(discharge_capacities)
    # convert capacity to percentage
    discharge_capacities /= discharge_capacities[0]
    return discharge_capacities


def find_regen_cycles(history, threshold=0.005):
    return [
        i + 1
        for i in range(len(history[1:]))
        if history[i + 1] - history[i] > threshold
    ]
