import os
import time
import torch
import numpy as np
import random

def make_file_path(file_path, is_pl = False):
    basepath = os.path.split(os.path.realpath(file_path))[0]
    fullpath = os.path.realpath(file_path)
    filename = os.path.split(os.path.realpath(file_path))[1]
    t = time.strftime("%b%d%H%M", time.localtime())
    os.system("mkdir -p _archives _outs")
    if not is_pl:
        os.system("mkdir -p _models/%s/best _models/%s/final" % (t, t))
    os.system("cp %s %s/_archives/%s.py" % (fullpath, basepath, t))
    modelpath = os.path.join(basepath, "_models", "%s.pt" % t)
    # plotpath = os.path.join(basepath, "_plots", "%s.png" % t)
    outpath = os.path.join(basepath, "_outs", "%s.out" % t)
    return modelpath, "", outpath, t, basepath

class Log():
    def __init__(self, outpath):
        self.outpath = outpath

    def __call__(self, *args):
        with open(self.outpath, 'a') as f:
            f.write(' '.join([str(arg) for arg in args]))
            f.write('\n')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True