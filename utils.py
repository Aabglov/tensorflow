import pickle
import os


def save(obj,dir_path,name,protocol=False):
    if protocol:
        with open(os.path.join(dir_path,name),"wb+") as f:
            pickle.dump(obj,f,protocol=protocol)
    else:
        with open(os.path.join(dir_path,name),"wb+") as f:
            pickle.dump(obj,f)

def load(dir_path,name):
    with open(os.path.join(dir_path,name),"rb") as f:
        return pickle.load(f)
