import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def reader(path):
    iters = []
    res_appared = False
    with open(path) as f:
        for line in f:
            li = line.split()
            if len(li) == 0:
                continue
            if li[0] == "[Iter]":
                if res_appared:
                    iters.append(int(li[2]))
                else:
                    iters.append(None)
                res_appared = False
            if li[0] == "[Res]":
                if float(li[2]) != 0:
                    res_appared = True
    return iters

def display(x, ylist, labels):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        })
    plt.rcParams["figure.figsize"] = (6, 3)

    fig, ax = plt.subplots()

    ## Plot
    for i, y in enumerate(ylist):
        ax.plot(x, y, markersize=7, lw=2, label=labels[i])

    ax.legend(fontsize=14)

    ## Ticks
    ax.tick_params(labelsize=13)

    ## Labels
    ax.set_xlabel('The value of $F_G$', fontsize=16)
    ax.set_ylabel('\# of iterations', fontsize=16)

    fig.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    path = "../Result/"

    matrices = []
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            if "_manual" in f:
                name = f.replace("_manual", "")
                matrices.append(name)

    li_iters = []
    for matrix in matrices:
        iters = reader(path+matrix+"_manual"+"/"+matrix+"_int_seq_20.txt")
        li_iters.append(iters)

    x = [i for i in range(20, 31)]

    display(x, li_iters, matrices);

    

