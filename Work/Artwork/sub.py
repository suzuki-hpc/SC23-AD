import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re

def reader(path):
    itr = None
    res_appared = False
    with open(path) as f:
        for line in f:
            li = line.split()
            if len(li) == 0:
                break
            if li[0] == "[Iter]":
                itr = int(li[2])
            if li[0] == "[Res]":
                res_appared = True
    if not res_appared:
        itr = None
    return itr

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
            if "_sub" in f:
                name = f.replace("_sub", "")
                matrices.append(name)

    x = [i for i in range(20, 31)]

    li_iters = []
    for matrix in matrices:
        iters = []
        for i in range(20, 31):
            itr = reader(path+matrix+"_sub"+"/"+matrix+"_int_seq_20_"+str(i)+".txt")
            iters.append(itr)
        print(iters)
        li_iters.append(iters)

    display(x, li_iters, matrices);

    

