import sys
import numpy as np
import matplotlib.pyplot as plt
import re

def reader(path):
    hist = [1.0]
    is_converged = True
    with open(path) as f:
        for line in f:
            if re.search(r'# \[\d\]', line):
                hist.append(float(line.split()[2]))
            elif line.split()[0] == "[Iter]" and line.split()[2] == "0":
                is_converged = False 
    return is_converged, hist

def display(d_x, d_y, f_x, f_y, i_x, i_y):
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Times"],
        })
    plt.rcParams["figure.figsize"] = (4, 4)

    fig, ax = plt.subplots()

    ## Plot
    ax.plot(d_x, d_y, marker="1", markersize=7, lw=2, linestyle="solid",  color="tab:cyan",   label="FP64")
    ax.plot(f_x, f_y, marker="+", markersize=7, lw=2, linestyle="dashed", color="tab:pink",   label="FP32")
    ax.plot(i_x, i_y, marker="x", markersize=7, lw=2, linestyle="dotted", color="tab:orange", label="INT")

    ax.legend(fontsize=14)

    ## Limit and yscale
    ax.set_xlim(left=0, right=max(d_x[-1],f_x[-1],i_x[-1]))
    ax.set_ylim(top=1.0, bottom=1.0e-10)
    ax.set_yscale("log")

    ## Ticks
    ax.tick_params(labelsize=13)

    ## Labels
    ax.set_xlabel("Number of the iterations", fontsize=16)
    ax.set_ylabel("Relative residual 2-norm", fontsize=16)

    fig.tight_layout()
    
    plt.show()

if __name__ == '__main__':
    path = "../Result/"
    period = ["5", "10", "20"]

    mm = sys.argv[1]
    period = sys.argv[2]
    smo = sys.argv[3]

    d_flag, d_hist = reader(path+mm+"/"+mm+"_fp64_"+smo+"_"+period+".txt")
    f_flag, f_hist = reader(path+mm+"/"+mm+"_fp32_"+smo+"_"+period+".txt")
    i_flag, i_hist = reader(path+mm+"/"+mm+"_int_"+smo+"_"+period+".txt")

    d_x = [i for i in range(len(d_hist))]
    f_x = [i for i in range(len(f_hist))]
    i_x = [i for i in range(len(i_hist))]

    display(d_x, d_hist, f_x, f_hist, i_x, i_hist)

    

