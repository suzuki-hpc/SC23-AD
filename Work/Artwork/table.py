import sys
import os
from tabulate import tabulate

def reader(path):
    imp = None
    with open(path) as f:
        for line in f:
            li = line.split()
            if li[0] == "#" and li[1] == "[Res]":
                exp = float(li[3])
            if li[0] == "[Time]":
                time = float(li[2])
            elif li[0] == "[Iter]":
                itr = int(li[2])
            elif li[0] == "[Res]":
                imp = float(li[2])
    if itr == 0:
        time = None
        itr = None
    return time, itr, imp, exp

if __name__ == '__main__':
    path = "../Result/"
    header = [
        "\nMatrix",
        "\nTime", "FP64\nItr.", "\nTime", "FP32\nItr.", "\nTime", "INT\nItr.",
        "\nTime", "FP64\nItr.", "\nTime", "FP32\nItr.", "\nTime", "INT\nItr.",
        "\nTime", "FP64\nItr.", "\nTime", "FP32\nItr.", "\nTime", "INT\nItr."]

    smo = sys.argv[1]

    matrices = []
    for f in os.listdir(path):
        if os.path.isdir(os.path.join(path, f)):
            if "_sub" not in f:
                matrices.append(f)

    for m in [5, 10, 20]:
        print("m = "+str(m))
        table = []
        for matrix in matrices:
            entry = [matrix]
            dtime, ditr, dimp, dexp = reader(path+matrix+"/"+matrix+"_fp64_"+smo+"_"+str(m)+".txt")
            ftime, fitr, fimp, fexp = reader(path+matrix+"/"+matrix+"_fp32_"+smo+"_"+str(m)+".txt")
            itime, iitr, iimp, iexp = reader(path+matrix+"/"+matrix+"_int_"+smo+"_"+str(m)+".txt")
            entry.append(dtime)
            entry.append(ditr)
            entry.append(ftime)
            entry.append(fitr)
            entry.append(itime)
            entry.append(iitr)

            table.append(entry)

        print(tabulate(table, headers=header, floatfmt=".2e"))
        print();

        
